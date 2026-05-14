# Copyright 2025-2026 Horizon RL Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Code sandbox toolkit using AWS Bedrock AgentCore Code Interpreter."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from typing import TYPE_CHECKING, Any

import botocore.exceptions
from aiolimiter import AsyncLimiter
from strands import tool

if TYPE_CHECKING:
    from strands_env.utils.aws import BotoClient

logger = logging.getLogger(__name__)


async def create_aio_client(
    region_name: str = "us-east-1",
    role_arn: str | None = None,
    max_pool_connections: int = 1024,
    connect_timeout: int = 120,
    read_timeout: int = 120,
) -> Any:
    """Create a shared async Bedrock AgentCore client.

    The returned aiobotocore client is already entered. Callers that create it
    are responsible for closing it with ``await client.__aexit__(None, None,
    None)`` when all toolkits using the client are done.

    Args:
        region_name: AWS region for the Bedrock AgentCore service.
        role_arn: Optional cross-account role to assume. Refreshable
            credentials are used so long-running jobs can continue starting new
            sessions after the first STS token expires.
        max_pool_connections: Maximum connections in the underlying aiohttp
            connection pool.
        connect_timeout: Connection timeout in seconds.
        read_timeout: Read timeout in seconds.

    Returns:
        An entered aiobotocore Bedrock AgentCore client.
    """
    import aiobotocore.session
    from botocore.config import Config

    config = Config(
        max_pool_connections=max_pool_connections,
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
        retries={"max_attempts": 3, "mode": "adaptive"},
    )

    if role_arn is not None:
        from aiobotocore.credentials import AioRefreshableCredentials

        async def refresh() -> dict[str, Any]:
            session = aiobotocore.session.get_session()
            async with session.create_client("sts", region_name=region_name) as sts:
                response = await sts.assume_role(
                    RoleArn=role_arn,
                    RoleSessionName="strands-env-code-interpreter",
                )
            credentials = response["Credentials"]
            logger.info(
                "Refreshed AgentCore role credentials: role=%s expiry=%s",
                role_arn,
                credentials["Expiration"].isoformat(),
            )
            return {
                "access_key": credentials["AccessKeyId"],
                "secret_key": credentials["SecretAccessKey"],
                "token": credentials["SessionToken"],
                "expiry_time": credentials["Expiration"].isoformat(),
            }

        initial = await refresh()
        refreshable = AioRefreshableCredentials.create_from_metadata(
            metadata=initial,
            refresh_using=refresh,
            method="sts-assume-role",
        )
        session = aiobotocore.session.AioSession()
        session._credentials = refreshable  # noqa: SLF001
        client = await session.create_client("bedrock-agentcore", region_name=region_name, config=config).__aenter__()
        client._strands_env_client_created_epoch = time.time()
        client._strands_env_credentials_expiration_epoch = datetime.fromisoformat(initial["expiry_time"]).timestamp()
        return client

    session = aiobotocore.session.get_session()
    client = await session.create_client("bedrock-agentcore", region_name=region_name, config=config).__aenter__()
    client._strands_env_client_created_epoch = time.time()
    return client


class CodeInterpreterQuotas:
    """Shared AWS quotas for Code Interpreter API operations.

    Notes:
        - Create one instance and pass it to all `CodeInterpreterToolkit` instances
        to enforce account-wide limits across concurrent sessions.
        - Manages three concerns:
            - Session semaphore: caps concurrent sessions (`session_concurrency`).
            - Rate limiters: caps API request initiation rate for start/invoke/stop
              (AWS TPS quotas) to prevent throttling errors.
            - Thread pool executor: sized to match `session_concurrency` so each session can
              have one in-flight blocking boto3 call without starving others.

    References:
        - AWS Bedrock AgentCore default quotas:
          https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/bedrock-agentcore-limits.html
    """

    DEFAULT_SESSION_CONCURRENCY = 1000
    DEFAULT_START_TPS = 30
    DEFAULT_INVOKE_TPS = 30
    DEFAULT_STOP_TPS = 30

    def __init__(
        self,
        session_concurrency: int = DEFAULT_SESSION_CONCURRENCY,
        start_tps: float = DEFAULT_START_TPS,
        invoke_tps: float = DEFAULT_INVOKE_TPS,
        stop_tps: float = DEFAULT_STOP_TPS,
    ):
        """Initialize a `CodeInterpreterQuotas` instance."""
        if session_concurrency < 1:
            raise ValueError("session_concurrency must be >= 1")
        if start_tps <= 0:
            raise ValueError("start_tps must be > 0")
        if invoke_tps <= 0:
            raise ValueError("invoke_tps must be > 0")
        if stop_tps <= 0:
            raise ValueError("stop_tps must be > 0")
        self.session_semaphore = asyncio.Semaphore(session_concurrency)
        self.start_limiter = AsyncLimiter(start_tps, time_period=1)
        self.invoke_limiter = AsyncLimiter(invoke_tps, time_period=1)
        self.stop_limiter = AsyncLimiter(stop_tps, time_period=1)
        self.executor = ThreadPoolExecutor(max_workers=session_concurrency)

    def to_thread(self, func: Any, /, *args: Any, **kwargs: Any) -> Any:
        """Run a blocking function in the quotas thread pool."""
        return asyncio.get_running_loop().run_in_executor(self.executor, partial(func, *args, **kwargs))

    async def acquire_session_slot(self) -> None:
        """Acquire one AgentCore session slot."""
        await self.session_semaphore.acquire()

    def release_session_slot(self) -> None:
        """Release one AgentCore session slot."""
        self.session_semaphore.release()

    async def acquire_start_slot(self) -> None:
        """Acquire one StartCodeInterpreterSession request slot."""
        await self.start_limiter.acquire()

    async def acquire_invoke_slot(self) -> None:
        """Acquire one InvokeCodeInterpreter request slot."""
        await self.invoke_limiter.acquire()

    async def acquire_stop_slot(self) -> None:
        """Acquire one StopCodeInterpreterSession request slot."""
        await self.stop_limiter.acquire()


class CodeInterpreterToolkit:
    """Code toolkit using AWS Bedrock AgentCore Code Interpreter.

    Notes:
        - Provides `execute_code` and `execute_command` tools for running Python code
          and shell commands in a sandboxed environment.
        - Uses a single shared agentcore session through session ID. Call
          `cleanup` when done to close the session.
    """

    CODE_INTERPRETER_ID = "aws.codeinterpreter.v1"

    def __init__(
        self,
        client: BotoClient | Any | None = None,
        session_name: str = "strands-env",
        quotas: CodeInterpreterQuotas | None = None,
        *,
        aio_client: Any | None = None,
        session_timeout_seconds: int = 3600,
    ):
        """Initialize a `CodeInterpreterToolkit` instance.

        Args:
            client: boto3 client for bedrock-agentcore service. Existing sync
                clients still run in the quota thread pool.
            session_name: Name for the code interpreter session.
            quotas: Shared quotas for rate limiting, session concurrency, and thread pool.
                Create one `CodeInterpreterQuotas` instance and pass it to all toolkit
                instances to enforce account-wide limits.
            aio_client: Optional aiobotocore client from `create_aio_client`.
                When supplied, AgentCore calls are awaited directly instead of
                being dispatched to the sync thread pool.
            session_timeout_seconds: AgentCore session timeout passed to
                StartCodeInterpreterSession.
        """
        if client is not None and aio_client is not None:
            raise ValueError("Pass client or aio_client, not both")
        resolved_client = aio_client if aio_client is not None else client
        if resolved_client is None:
            raise ValueError("CodeInterpreterToolkit requires client or aio_client")
        self.session_name = session_name
        self.client = resolved_client
        self._client = resolved_client
        self._use_threadpool = aio_client is None
        self.session_id: str | None = None
        self.session_timeout_seconds = session_timeout_seconds
        self.quotas = quotas or CodeInterpreterQuotas()
        self._session_lock = asyncio.Lock()

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        """Await `value` if it is awaitable, otherwise return it."""
        if inspect.isawaitable(value):
            return await value
        return value

    @staticmethod
    async def _iter_stream(stream: Any) -> Any:
        """Iterate over either an async or sync AgentCore event stream."""
        if hasattr(stream, "__aiter__"):
            async for event in stream:
                yield event
            return
        for event in stream:
            yield event

    async def _call_client(self, operation: str, **kwargs: Any) -> Any:
        """Call a sync or async AgentCore client operation."""
        method = getattr(self._client, operation)
        if self._use_threadpool:
            result = await self.quotas.to_thread(method, **kwargs)
        else:
            result = method(**kwargs)
        return await self._maybe_await(result)

    @staticmethod
    def _raise_event_stream_error(key: str, payload: dict[str, Any]) -> None:
        """Raise an AgentCore EventStream error as a botocore `ClientError`."""
        code = key[:1].upper() + key[1:]
        message = str(payload.get("message") or payload.get("Message") or key)
        raise botocore.exceptions.ClientError(
            error_response={"Error": {"Code": code, "Message": message}},
            operation_name="InvokeCodeInterpreter",
        )

    async def start_session(self) -> None:
        """Start a code interpreter session if not already started (async, thread-safe)."""
        if self.session_id is None:
            async with self._session_lock:
                # Double-check after acquiring lock as another coroutine may have set it
                if self.session_id is not None:
                    return  # type: ignore[unreachable]

                await self.quotas.acquire_session_slot()
                await self.quotas.acquire_start_slot()
                try:
                    response = await self._call_client(
                        "start_code_interpreter_session",
                        codeInterpreterIdentifier=self.CODE_INTERPRETER_ID,
                        name=self.session_name,
                        sessionTimeoutSeconds=self.session_timeout_seconds,
                    )
                except Exception:
                    self.quotas.release_session_slot()
                    raise
                self.session_id = response["sessionId"]

    async def invoke(self, name: str, arguments: dict[str, Any]) -> str:
        """Invoke the code interpreter and return parsed response."""
        await self.start_session()
        await self.quotas.acquire_invoke_slot()
        response = await self._call_client(
            "invoke_code_interpreter",
            codeInterpreterIdentifier=self.CODE_INTERPRETER_ID,
            sessionId=self.session_id,
            name=name,
            arguments=arguments,
        )
        # Parse the `EventStream` response from `invoke_code_interpreter`.
        stream = response.get("stream")
        if stream is not None:
            async for event in self._iter_stream(stream):
                if "result" in event:
                    content = event["result"].get("content", [])
                    if isinstance(content, list):
                        texts = [c.get("text", "") for c in content if c.get("type") == "text"]
                        return "\n".join(texts) if texts else str(content)
                    return str(content)

                for key in (
                    "accessDeniedException",
                    "conflictException",
                    "internalServerException",
                    "resourceNotFoundException",
                    "serviceQuotaExceededException",
                    "throttlingException",
                    "validationException",
                ):
                    if key in event:
                        payload = event[key] if isinstance(event[key], dict) else {"message": event[key]}
                        self._raise_event_stream_error(key, payload)

        return "No result returned."

    @tool
    async def execute_code(self, code: str) -> str:
        """Execute Python code and return the result.

        Args:
            code: The Python code to execute.

        Returns:
            Execution output text or error message.
        """
        return await self.invoke("executeCode", {"code": code, "language": "python"})

    @tool
    async def execute_command(self, command: str) -> str:
        """Execute a shell command and return the result.

        Args:
            command: The shell command to execute.

        Returns:
            Execution output text or error message.
        """
        return await self.invoke("executeCommand", {"command": command})

    async def cleanup(self) -> None:
        """Clean up code interpreter session."""
        if self.session_id:
            await self.quotas.acquire_stop_slot()
            try:
                await self._call_client(
                    "stop_code_interpreter_session",
                    codeInterpreterIdentifier=self.CODE_INTERPRETER_ID,
                    sessionId=self.session_id,
                )
            except Exception:
                pass  # Ignore cleanup errors
            finally:
                self.quotas.release_session_slot()
            self.session_id = None
