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

"""Reusable AgentCore Code Interpreter session pool."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import botocore.exceptions

from strands_env.tools.code_interpreter import CodeInterpreterQuotas, CodeInterpreterToolkit, create_aio_client

logger = logging.getLogger(__name__)


class AgentCoreSessionState(Enum):
    """Lifecycle states for one pooled AgentCore session."""

    SPAWNING = "spawning"
    READY = "ready"
    DRAINING = "draining"
    RECYCLING = "recycling"
    DEAD = "dead"


@dataclass(frozen=True)
class AgentCorePoolConfig:
    """Configuration for `AgentCorePool`."""

    target_size: int = 16
    max_pool_size: int = 1000
    max_in_flight_per_session: int = 1
    session_timeout_seconds: int = 3600
    drain_before_expiry_secs: float = 480.0
    recycle_before_expiry_secs: float = 120.0
    monitor_interval_secs: float = 2.0
    spawn_concurrency: int = 4
    session_name_prefix: str = "strands-env-agentcore"

    def __post_init__(self) -> None:
        """Validate pool invariants."""
        if self.target_size < 1:
            raise ValueError("target_size must be >= 1")
        if self.max_pool_size < self.target_size:
            raise ValueError("max_pool_size must be >= target_size")
        if self.max_in_flight_per_session < 1:
            raise ValueError("max_in_flight_per_session must be >= 1")
        if self.session_timeout_seconds <= 0:
            raise ValueError("session_timeout_seconds must be > 0")
        if self.recycle_before_expiry_secs < 0:
            raise ValueError("recycle_before_expiry_secs must be >= 0")
        if self.drain_before_expiry_secs <= self.recycle_before_expiry_secs:
            raise ValueError("drain_before_expiry_secs must be greater than recycle_before_expiry_secs")
        if self.drain_before_expiry_secs >= self.session_timeout_seconds:
            raise ValueError("drain_before_expiry_secs must be less than session_timeout_seconds")
        if self.monitor_interval_secs <= 0:
            raise ValueError("monitor_interval_secs must be > 0")
        if self.spawn_concurrency < 1:
            raise ValueError("spawn_concurrency must be >= 1")


@dataclass
class AgentCoreSession:
    """One pooled AgentCore Code Interpreter session."""

    toolkit: CodeInterpreterToolkit
    state: AgentCoreSessionState = AgentCoreSessionState.SPAWNING
    created_monotonic: float = field(default_factory=time.monotonic)
    in_flight: int = 0
    invoke_count: int = 0

    @property
    def id(self) -> str | None:
        """Return the AgentCore session id when started."""
        return self.toolkit.session_id


@dataclass(frozen=True)
class AgentCorePoolStats:
    """Snapshot of pool state."""

    target_size: int
    total_sessions: int
    spawning: int
    ready: int
    draining: int
    recycling: int
    dead: int
    in_flight: int
    spawned_total: int
    spawn_failures_total: int
    recycled_total: int
    session_dead_retries_total: int
    waits_total: int

    def as_dict(self) -> dict[str, int]:
        """Return stats as a plain dictionary."""
        return {
            "agentcore_pool_target_size": self.target_size,
            "agentcore_pool_total_sessions": self.total_sessions,
            "agentcore_pool_spawning": self.spawning,
            "agentcore_pool_ready": self.ready,
            "agentcore_pool_draining": self.draining,
            "agentcore_pool_recycling": self.recycling,
            "agentcore_pool_dead": self.dead,
            "agentcore_pool_in_flight": self.in_flight,
            "agentcore_pool_spawned_total": self.spawned_total,
            "agentcore_pool_spawn_failures_total": self.spawn_failures_total,
            "agentcore_pool_recycled_total": self.recycled_total,
            "agentcore_pool_session_dead_retries_total": self.session_dead_retries_total,
            "agentcore_pool_waits_total": self.waits_total,
        }


@dataclass
class _Counters:
    spawned_total: int = 0
    spawn_failures_total: int = 0
    recycled_total: int = 0
    session_dead_retries_total: int = 0
    waits_total: int = 0


def _is_session_dead(exc: BaseException) -> bool:
    """Return whether `exc` means the referenced AgentCore session is dead."""
    if not isinstance(exc, botocore.exceptions.ClientError):
        return False
    error = exc.response.get("Error", {})
    code = str(error.get("Code", ""))
    message = (str(error.get("Message", "")) + " " + str(exc)).lower()
    if code == "ValidationException" and "not active" in message:
        return True
    if code == "ConflictException":
        return (
            "terminated on user request" in message
            or "exceeded the configured session timeout" in message
            or "ended before the request could complete" in message
        )
    return False


def _is_non_retryable_spawn_error(exc: BaseException) -> bool:
    """Return whether a session-spawn error cannot be fixed by retrying."""
    if not isinstance(exc, botocore.exceptions.ClientError):
        return False
    error = exc.response.get("Error", {})
    code = str(error.get("Code", ""))
    return code in {
        "AccessDenied",
        "AccessDeniedException",
        "InvalidClientTokenId",
        "InvalidSignatureException",
        "UnauthorizedOperation",
        "UnrecognizedClientException",
    }


class AgentCorePool:
    """Pool and scheduler for AgentCore Code Interpreter sessions."""

    def __init__(
        self,
        *,
        config: AgentCorePoolConfig | None = None,
        client: Any | None = None,
        aio_client: Any | None = None,
        quotas: CodeInterpreterQuotas | None = None,
        region_name: str = "us-east-1",
        role_arn: str | None = None,
        max_pool_connections: int = 1024,
        connect_timeout: int = 120,
        read_timeout: int = 120,
    ) -> None:
        """Initialize an `AgentCorePool` instance."""
        if client is not None and aio_client is not None:
            raise ValueError("Pass client or aio_client, not both")
        self.config = config or AgentCorePoolConfig()
        self._provided_client = client
        self._provided_aio_client = aio_client
        self._client = aio_client if aio_client is not None else client
        self._owns_client = client is None and aio_client is None
        self._region_name = region_name
        self._role_arn = role_arn
        self._max_pool_connections = max_pool_connections
        self._connect_timeout = connect_timeout
        self._read_timeout = read_timeout
        self._quotas = quotas or CodeInterpreterQuotas()
        self._sessions: list[AgentCoreSession] = []
        self._lock = asyncio.Lock()
        self._maintain_lock = asyncio.Lock()
        self._capacity_event = asyncio.Event()
        self._capacity_event.set()
        self._closed = False
        self._fatal_spawn_error: BaseException | None = None
        self._monitor_task: asyncio.Task[None] | None = None
        self._spawn_counter = 0
        self._counters = _Counters()

    async def start(self) -> None:
        """Start the pool monitor and begin filling toward target size."""
        async with self._lock:
            if self._closed:
                raise RuntimeError("AgentCorePool is closed")
            if self._client is None:
                self._client = await create_aio_client(
                    region_name=self._region_name,
                    role_arn=self._role_arn,
                    max_pool_connections=self._max_pool_connections,
                    connect_timeout=self._connect_timeout,
                    read_timeout=self._read_timeout,
                )
            if self._monitor_task is None:
                self._monitor_task = asyncio.create_task(self._monitor_loop(), name="agentcore-pool-monitor")
        await self.maintain_once()

    async def prewarm(self, target: int | None = None) -> int:
        """Wait until `target` sessions are ready, clamped to pool target size."""
        await self.start()
        wanted = self.config.target_size if target is None else min(target, self.config.target_size)
        wanted = max(0, wanted)
        while True:
            ready = self.stats().ready
            if ready >= wanted:
                return ready
            self._raise_if_fatal_spawn_error()
            self._capacity_event.clear()
            await self.maintain_once()
            await self._capacity_event.wait()

    async def invoke(self, code: str, language: str = "python") -> str:
        """Run code on a vacant AgentCore session, retrying once if a session dies."""
        await self.start()
        attempts = 0
        last_dead_error: botocore.exceptions.ClientError | None = None
        while attempts < 2:
            attempts += 1
            session = await self._acquire()
            session_dead = False
            try:
                return await session.toolkit.invoke("executeCode", {"code": code, "language": language})
            except botocore.exceptions.ClientError as exc:
                if not _is_session_dead(exc):
                    raise
                last_dead_error = exc
                session_dead = True
                await self._mark_session(session, AgentCoreSessionState.DEAD)
                async with self._lock:
                    self._counters.session_dead_retries_total += 1
            finally:
                await self._release(session)
            if session_dead:
                await self.maintain_once()
        if last_dead_error is not None:
            raise last_dead_error
        raise RuntimeError("AgentCorePool invoke failed without an error")

    async def maintain_once(self) -> None:
        """Run one maintenance pass: sweep, recycle, and top up sessions."""
        async with self._maintain_lock:
            async with self._lock:
                if self._closed:
                    return
                self._sweep_states_locked()
            await self._reap_recycled()
            await self._top_up_to_target()

    def stats(self) -> AgentCorePoolStats:
        """Return a point-in-time pool stats snapshot."""
        counts = {state: 0 for state in AgentCoreSessionState}
        in_flight = 0
        for session in self._sessions:
            counts[session.state] += 1
            in_flight += session.in_flight
        return AgentCorePoolStats(
            target_size=self.config.target_size,
            total_sessions=len(self._sessions),
            spawning=counts[AgentCoreSessionState.SPAWNING],
            ready=counts[AgentCoreSessionState.READY],
            draining=counts[AgentCoreSessionState.DRAINING],
            recycling=counts[AgentCoreSessionState.RECYCLING],
            dead=counts[AgentCoreSessionState.DEAD],
            in_flight=in_flight,
            spawned_total=self._counters.spawned_total,
            spawn_failures_total=self._counters.spawn_failures_total,
            recycled_total=self._counters.recycled_total,
            session_dead_retries_total=self._counters.session_dead_retries_total,
            waits_total=self._counters.waits_total,
        )

    async def shutdown(self) -> None:
        """Stop the monitor, clean up sessions, and close an owned async client."""
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            monitor_task = self._monitor_task
            self._monitor_task = None
            sessions = list(self._sessions)
            self._sessions.clear()
            client = self._client
            self._client = None
            self._capacity_event.set()

        if monitor_task is not None:
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

        await asyncio.gather(*(self._safe_cleanup(session) for session in sessions), return_exceptions=True)
        if self._owns_client and client is not None and hasattr(client, "__aexit__"):
            await client.__aexit__(None, None, None)

    async def _acquire(self) -> AgentCoreSession:
        while True:
            async with self._lock:
                if self._closed:
                    raise RuntimeError("AgentCorePool is closed")
                self._sweep_states_locked()
                session = self._pick_best_locked()
                if session is not None:
                    session.in_flight += 1
                    session.invoke_count += 1
                    return session
                if self._fatal_spawn_error is not None:
                    self._raise_fatal_spawn_error_locked()
                self._counters.waits_total += 1
                self._capacity_event.clear()
            await self.maintain_once()
            await self._capacity_event.wait()

    async def _release(self, session: AgentCoreSession) -> None:
        async with self._lock:
            if session.in_flight > 0:
                session.in_flight -= 1
            if session.state is AgentCoreSessionState.DRAINING and session.in_flight == 0:
                session.state = AgentCoreSessionState.RECYCLING
            self._capacity_event.set()

    async def _mark_session(self, session: AgentCoreSession, state: AgentCoreSessionState) -> None:
        async with self._lock:
            if session in self._sessions:
                session.state = state
            self._capacity_event.set()

    def _pick_best_locked(self) -> AgentCoreSession | None:
        candidates = [
            session
            for session in self._sessions
            if session.state is AgentCoreSessionState.READY
            and session.in_flight < self.config.max_in_flight_per_session
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda session: (session.in_flight, -self._remaining_secs(session)))

    def _remaining_secs(self, session: AgentCoreSession) -> float:
        age = time.monotonic() - session.created_monotonic
        return self.config.session_timeout_seconds - age

    def _sweep_states_locked(self) -> None:
        for session in self._sessions:
            if session.state not in (AgentCoreSessionState.READY, AgentCoreSessionState.DRAINING):
                continue
            remaining = self._remaining_secs(session)
            if remaining <= self.config.recycle_before_expiry_secs:
                session.state = AgentCoreSessionState.RECYCLING
            elif session.state is AgentCoreSessionState.READY and remaining <= self.config.drain_before_expiry_secs:
                session.state = AgentCoreSessionState.DRAINING
            if session.state is AgentCoreSessionState.DRAINING and session.in_flight == 0:
                session.state = AgentCoreSessionState.RECYCLING

    async def _reap_recycled(self) -> None:
        async with self._lock:
            to_recycle = [
                session
                for session in self._sessions
                if session.state in (AgentCoreSessionState.RECYCLING, AgentCoreSessionState.DEAD)
                and session.in_flight == 0
            ]
            if not to_recycle:
                return
            self._sessions = [session for session in self._sessions if session not in to_recycle]
            self._counters.recycled_total += len(to_recycle)
            self._capacity_event.set()
        await asyncio.gather(*(self._safe_cleanup(session) for session in to_recycle), return_exceptions=True)

    async def _top_up_to_target(self) -> None:
        async with self._lock:
            if self._closed or self._fatal_spawn_error is not None:
                return
            active = sum(
                1
                for session in self._sessions
                if session.state in (AgentCoreSessionState.SPAWNING, AgentCoreSessionState.READY)
            )
            deficit = min(
                self.config.target_size - active,
                self.config.max_pool_size - len(self._sessions),
                self.config.spawn_concurrency,
            )
        if deficit <= 0:
            return
        await asyncio.gather(*(self._spawn_one() for _ in range(deficit)), return_exceptions=True)

    async def _spawn_one(self) -> None:
        async with self._lock:
            if self._closed or self._client is None:
                return
            self._spawn_counter += 1
            toolkit = CodeInterpreterToolkit(
                client=self._provided_client,
                aio_client=self._client if self._provided_client is None else None,
                session_name=f"{self.config.session_name_prefix}-{self._spawn_counter}",
                quotas=self._quotas,
                session_timeout_seconds=self.config.session_timeout_seconds,
            )
            session = AgentCoreSession(toolkit=toolkit)
            self._sessions.append(session)

        try:
            await toolkit.start_session()
        except Exception as exc:  # noqa: BLE001
            is_fatal = _is_non_retryable_spawn_error(exc)
            log = logger.error if is_fatal else logger.warning
            log("AgentCore session spawn failed: %s: %s", type(exc).__name__, exc)
            async with self._lock:
                if is_fatal and self._fatal_spawn_error is None:
                    self._fatal_spawn_error = exc
                self._counters.spawn_failures_total += 1
                if session in self._sessions:
                    self._sessions.remove(session)
                self._capacity_event.set()
            await self._safe_cleanup(session)
            return

        cleanup_after_spawn = False
        async with self._lock:
            if self._closed or session not in self._sessions:
                session.state = AgentCoreSessionState.RECYCLING
                cleanup_after_spawn = True
            else:
                session.created_monotonic = time.monotonic()
                session.state = AgentCoreSessionState.READY
                self._counters.spawned_total += 1
            self._capacity_event.set()
        if cleanup_after_spawn:
            await self._safe_cleanup(session)

    async def _monitor_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self.config.monitor_interval_secs)
                await self.maintain_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("AgentCorePool monitor iteration failed")

    def _raise_if_fatal_spawn_error(self) -> None:
        if self._fatal_spawn_error is not None:
            self._raise_fatal_spawn_error_locked()

    def _raise_fatal_spawn_error_locked(self) -> None:
        raise RuntimeError("AgentCorePool cannot spawn AgentCore sessions") from self._fatal_spawn_error

    @staticmethod
    async def _safe_cleanup(session: AgentCoreSession) -> None:
        try:
            await session.toolkit.cleanup()
        except Exception as exc:  # noqa: BLE001
            logger.warning("AgentCore session cleanup failed: %s: %s", type(exc).__name__, exc)
