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

"""Unit tests for `AgentCorePool`."""

from __future__ import annotations

import asyncio

import botocore.exceptions
import pytest

from strands_env.tools import AgentCorePool, AgentCorePoolConfig, AgentCoreSessionState, CodeInterpreterQuotas
from strands_env.tools.agent_core_pool import _is_non_retryable_spawn_error, _is_session_dead


class _AsyncStream:
    def __init__(self, events):
        self._events = list(events)
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._events):
            raise StopAsyncIteration
        event = self._events[self._index]
        self._index += 1
        return event


class _AgentCoreClientStub:
    def __init__(self, *, invoke_latency: float = 0.0, max_in_flight_per_session: int = 1) -> None:
        self.invoke_latency = invoke_latency
        self.max_in_flight_per_session = max_in_flight_per_session
        self.starts = 0
        self.stops = 0
        self.invokes = 0
        self.conflicts = 0
        self.max_session_in_flight = 0
        self.fail_next_session_dead = False
        self.start_error: botocore.exceptions.ClientError | None = None
        self._session_in_flight: dict[str, int] = {}

    async def start_code_interpreter_session(self, **_kwargs):
        self.starts += 1
        if self.start_error is not None:
            raise self.start_error
        session_id = f"session-{self.starts}"
        self._session_in_flight[session_id] = 0
        return {"sessionId": session_id}

    async def invoke_code_interpreter(self, **kwargs):
        session_id = kwargs["sessionId"]
        if self.fail_next_session_dead:
            self.fail_next_session_dead = False
            raise _client_error("ValidationException", f"Code interpreter session {session_id} is not active")
        in_flight = self._session_in_flight.get(session_id)
        if in_flight is None:
            raise _client_error("ValidationException", f"Code interpreter session {session_id} is not active")
        if in_flight >= self.max_in_flight_per_session:
            self.conflicts += 1
            raise _client_error("ConflictException", f"session {session_id} already has an in-flight invoke")

        self._session_in_flight[session_id] = in_flight + 1
        self.max_session_in_flight = max(self.max_session_in_flight, self._session_in_flight[session_id])
        try:
            self.invokes += 1
            await asyncio.sleep(self.invoke_latency)
            text = f"ok:{session_id}:{kwargs['arguments']['code']}"
            return {"stream": _AsyncStream([{"result": {"content": [{"type": "text", "text": text}]}}])}
        finally:
            self._session_in_flight[session_id] -= 1

    async def stop_code_interpreter_session(self, **kwargs):
        self.stops += 1
        self._session_in_flight.pop(kwargs["sessionId"], None)
        return {}


def _client_error(
    code: str,
    message: str,
    *,
    operation_name: str = "InvokeCodeInterpreter",
) -> botocore.exceptions.ClientError:
    return botocore.exceptions.ClientError(
        error_response={"Error": {"Code": code, "Message": message}},
        operation_name=operation_name,
    )


def _pool(client: _AgentCoreClientStub, **config_kwargs) -> AgentCorePool:
    config = AgentCorePoolConfig(
        target_size=config_kwargs.pop("target_size", 2),
        monitor_interval_secs=0.05,
        drain_before_expiry_secs=config_kwargs.pop("drain_before_expiry_secs", 0.5),
        recycle_before_expiry_secs=config_kwargs.pop("recycle_before_expiry_secs", 0.2),
        session_timeout_seconds=config_kwargs.pop("session_timeout_seconds", 2),
        **config_kwargs,
    )
    quotas = CodeInterpreterQuotas(start_tps=1000, invoke_tps=1000, stop_tps=1000)
    return AgentCorePool(config=config, aio_client=client, quotas=quotas)


class TestSessionDeadClassification:
    def test_validation_not_active_is_session_dead(self):
        assert _is_session_dead(_client_error("ValidationException", "Code interpreter session abc is not active"))

    def test_timeout_conflict_is_session_dead(self):
        assert _is_session_dead(_client_error("ConflictException", "ended before the request could complete"))

    def test_quota_error_is_not_session_dead(self):
        assert not _is_session_dead(_client_error("ServiceQuotaExceededException", "quota"))


class TestSpawnErrorClassification:
    def test_access_denied_is_non_retryable_spawn_error(self):
        assert _is_non_retryable_spawn_error(
            _client_error("AccessDeniedException", "denied", operation_name="StartCodeInterpreterSession")
        )

    def test_quota_error_is_retryable_spawn_error(self):
        assert not _is_non_retryable_spawn_error(
            _client_error("ServiceQuotaExceededException", "quota", operation_name="StartCodeInterpreterSession")
        )


class TestAgentCorePool:
    async def test_prewarm_clamps_to_target_size(self):
        client = _AgentCoreClientStub()
        pool = _pool(client, target_size=2)
        try:
            ready = await asyncio.wait_for(pool.prewarm(target=8), timeout=1)
            assert ready == 2
            assert pool.stats().ready == 2
            assert client.starts == 2
        finally:
            await pool.shutdown()

    async def test_invoke_routes_to_ready_session(self):
        client = _AgentCoreClientStub()
        pool = _pool(client, target_size=2)
        try:
            await pool.prewarm()
            result = await pool.invoke("print(1)")
            assert result.startswith("ok:session-")
            assert result.endswith(":print(1)")
            assert client.invokes == 1
        finally:
            await pool.shutdown()

    async def test_pool_serializes_per_session_capacity(self):
        client = _AgentCoreClientStub(invoke_latency=0.02, max_in_flight_per_session=1)
        pool = _pool(client, target_size=2, max_in_flight_per_session=1)
        try:
            await pool.prewarm()
            results = await asyncio.gather(*(pool.invoke(f"sample-{index}") for index in range(20)))
            assert len(results) == 20
            assert client.conflicts == 0
            assert client.max_session_in_flight == 1
            assert client.invokes == 20
        finally:
            await pool.shutdown()

    async def test_concurrent_maintenance_does_not_overspawn(self):
        client = _AgentCoreClientStub(invoke_latency=0.01)
        pool = _pool(client, target_size=2, spawn_concurrency=2)
        try:
            await asyncio.gather(*(pool.maintain_once() for _ in range(50)))

            assert pool.stats().ready == 2
            assert client.starts == 2
        finally:
            await pool.shutdown()

    async def test_session_dead_retries_once_on_fresh_session(self):
        client = _AgentCoreClientStub()
        client.fail_next_session_dead = True
        pool = _pool(client, target_size=2)
        try:
            await pool.prewarm()
            result = await pool.invoke("recover")
            assert result.endswith(":recover")
            assert pool.stats().session_dead_retries_total == 1
            assert pool.stats().dead == 0
            assert pool.stats().recycled_total == 1
            assert client.invokes == 1
        finally:
            await pool.shutdown()

    async def test_aged_session_recycles_and_replaces(self):
        client = _AgentCoreClientStub()
        pool = _pool(client, target_size=1)
        try:
            await pool.prewarm()
            original = pool._sessions[0]
            original.created_monotonic -= 10
            await pool.maintain_once()
            await pool.prewarm()

            assert original not in pool._sessions
            assert client.stops >= 1
            assert client.starts >= 2
            assert pool.stats().ready == 1
        finally:
            await pool.shutdown()

    async def test_draining_sessions_do_not_count_as_ready_capacity(self):
        client = _AgentCoreClientStub()
        pool = _pool(client, target_size=2)
        try:
            await pool.prewarm()
            draining = pool._sessions[0]
            draining.state = AgentCoreSessionState.DRAINING
            await pool.maintain_once()
            await pool.prewarm()

            assert pool.stats().ready == 2
            assert client.starts >= 3
        finally:
            await pool.shutdown()

    async def test_prewarm_fails_fast_on_non_retryable_spawn_error(self):
        client = _AgentCoreClientStub()
        client.start_error = _client_error(
            "AccessDeniedException",
            "not authorized",
            operation_name="StartCodeInterpreterSession",
        )
        pool = _pool(client, target_size=1, spawn_concurrency=1)
        try:
            with pytest.raises(RuntimeError, match="cannot spawn AgentCore sessions"):
                await asyncio.wait_for(pool.prewarm(), timeout=1)

            assert client.starts == 1
            assert pool.stats().spawn_failures_total == 1
            assert pool.stats().ready == 0
        finally:
            await pool.shutdown()
