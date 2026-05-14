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

"""Unit tests for `CodeTestCaseHarnessReward`."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

import pytest

from strands_env.core.types import Action, Observation, StepResult, TaskContext, TerminationReason
from strands_env.rewards.code_test_case_harness_reward import _RESULTS_MARKER, CodeTestCaseHarnessReward


class ToolkitStub:
    """Minimal CodeInterpreterToolkit test double."""

    def __init__(self, response: str | None = None, error: BaseException | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.cleaned = False

    async def invoke(self, name: str, arguments: dict[str, Any]) -> str:
        self.calls.append((name, arguments))
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response

    async def cleanup(self) -> None:
        self.cleaned = True


class PoolStub:
    """Minimal AgentCorePool test double."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []

    async def invoke(self, code: str, language: str = "python") -> str:
        self.calls.append((code, language))
        return self.response


def _harness_output(per_test: list[dict[str, Any]]) -> str:
    return f"prelude\n{_RESULTS_MARKER}\n{json.dumps({'per_test': per_test})}\n"


def _make_action(inputs: list[str], outputs: list[str]) -> Action:
    return Action(
        message="solve",
        task_context=TaskContext(ground_truth={"inputs": inputs, "outputs": outputs}, conversation_history=[]),
    )


def _make_step(response: str | None) -> StepResult:
    content = [] if response is None else [{"text": response}]
    return StepResult(
        observation=Observation(messages=[{"role": "assistant", "content": content}], metrics={}),
        reward=None,
        termination_reason=TerminationReason.NOT_TERMINATED,
    )


async def test_reward_uses_one_invoke_for_multiple_tests() -> None:
    toolkit = ToolkitStub(_harness_output([{"passed": True}, {"passed": True}]))
    reward = CodeTestCaseHarnessReward(toolkit=toolkit, binary_reward=True)

    result = await reward.compute(
        _make_action(["1\n", "2\n"], ["1\n", "2\n"]),
        _make_step("```python\nprint(input())\n```"),
    )

    assert result.reward == 1.0
    assert result.info["passed"] == 2
    assert result.info["reward_masked"] is False
    assert len(toolkit.calls) == 1
    name, arguments = toolkit.calls[0]
    assert name == "executeCode"
    assert arguments["language"] == "python"


async def test_reward_can_use_agent_core_pool() -> None:
    pool = PoolStub(_harness_output([{"passed": True}]))
    reward = CodeTestCaseHarnessReward(agent_core_pool=pool)

    result = await reward.compute(
        _make_action(["1\n"], ["1\n"]),
        _make_step("```python\nprint(input())\n```"),
    )

    assert result.reward == 1.0
    assert pool.calls
    assert pool.calls[0][1] == "python"


def test_reward_rejects_ambiguous_execution_backends() -> None:
    with pytest.raises(ValueError, match="agent_core_pool or client"):
        CodeTestCaseHarnessReward(agent_core_pool=PoolStub("unused"), client=object())

    with pytest.raises(ValueError, match="toolkit or client"):
        CodeTestCaseHarnessReward(toolkit=ToolkitStub("unused"), client=object())


async def test_reward_can_capture_full_test_results() -> None:
    per_test = [{"passed": True, "test_case": index} for index in range(7)]
    toolkit = ToolkitStub(_harness_output(per_test))
    reward = CodeTestCaseHarnessReward(toolkit=toolkit, test_results_limit=None)

    result = await reward.compute(
        _make_action([f"{index}\n" for index in range(7)], [f"{index}\n" for index in range(7)]),
        _make_step("```python\nprint(input())\n```"),
    )

    assert result.info["test_results_preview"] == per_test


async def test_no_code_block_is_not_masked() -> None:
    toolkit = ToolkitStub("unused")
    reward = CodeTestCaseHarnessReward(toolkit=toolkit)

    result = await reward.compute(_make_action(["1\n"], ["1\n"]), _make_step("no code here"))

    assert result.reward == 0.0
    assert result.info["reason"] == "no_code_block_found"
    assert result.info["reward_masked"] is False
    assert toolkit.calls == []


async def test_malformed_harness_output_masks_reward() -> None:
    toolkit = ToolkitStub("sandbox output without marker")
    reward = CodeTestCaseHarnessReward(toolkit=toolkit)

    result = await reward.compute(_make_action(["1\n"], ["1\n"]), _make_step("```python\nprint(1)\n```"))

    assert result.reward == 0.0
    assert result.info["reward_masked"] is True
    assert result.info["reward_masked_reason"] == "malformed_harness_output"


async def test_invoke_exception_masks_reward() -> None:
    toolkit = ToolkitStub(error=RuntimeError("agentcore unavailable"))
    reward = CodeTestCaseHarnessReward(toolkit=toolkit)

    result = await reward.compute(_make_action(["1\n"], ["1\n"]), _make_step("```python\nprint(1)\n```"))

    assert result.reward == 0.0
    assert result.info["reward_masked"] is True
    assert result.info["reward_masked_reason"] == "harness_invoke_failed"
    assert result.info["last_error_type"] == "RuntimeError"


def test_built_harness_runs_all_tests_locally() -> None:
    reward = CodeTestCaseHarnessReward(toolkit=ToolkitStub("unused"))
    harness = reward.build_harness(
        "print(int(input()) * 2)",
        ["2\n", "3\n"],
        ["4\n", "6.0\n"],
    )

    completed = subprocess.run(
        [sys.executable, "-c", harness],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )

    marker_index = completed.stdout.rfind(_RESULTS_MARKER)
    assert marker_index >= 0
    payload_line = completed.stdout[marker_index + len(_RESULTS_MARKER) :].strip().split("\n", 1)[0]
    payload = json.loads(payload_line)
    assert [result["passed"] for result in payload["per_test"]] == [True, True]


def test_built_harness_records_failed_test_locally() -> None:
    reward = CodeTestCaseHarnessReward(toolkit=ToolkitStub("unused"))
    harness = reward.build_harness("print('wrong')", ["1\n"], ["right\n"])

    completed = subprocess.run(
        [sys.executable, "-c", harness],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )

    marker_index = completed.stdout.rfind(_RESULTS_MARKER)
    payload_line = completed.stdout[marker_index + len(_RESULTS_MARKER) :].strip().split("\n", 1)[0]
    payload = json.loads(payload_line)
    assert payload["per_test"][0]["passed"] is False
    assert payload["all_pass"] is False


def test_built_harness_rejects_nonzero_exit_even_when_stdout_matches() -> None:
    reward = CodeTestCaseHarnessReward(toolkit=ToolkitStub("unused"))
    harness = reward.build_harness("print('right')\nraise RuntimeError('boom')", ["1\n"], ["right\n"])

    completed = subprocess.run(
        [sys.executable, "-c", harness],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )

    marker_index = completed.stdout.rfind(_RESULTS_MARKER)
    payload_line = completed.stdout[marker_index + len(_RESULTS_MARKER) :].strip().split("\n", 1)[0]
    payload = json.loads(payload_line)
    result = payload["per_test"][0]
    assert result["passed"] is False
    assert result["returncode"] != 0
    assert "RuntimeError" in result["stderr_head"]
    assert payload["all_pass"] is False


def test_built_harness_accepts_decimal_tolerance_locally() -> None:
    reward = CodeTestCaseHarnessReward(toolkit=ToolkitStub("unused"))
    harness = reward.build_harness("print('1.916666666666667')", ["1\n"], ["1.91666666666666674068\n"])

    completed = subprocess.run(
        [sys.executable, "-c", harness],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )

    marker_index = completed.stdout.rfind(_RESULTS_MARKER)
    payload_line = completed.stdout[marker_index + len(_RESULTS_MARKER) :].strip().split("\n", 1)[0]
    payload = json.loads(payload_line)
    assert payload["per_test"][0]["passed"] is True
    assert payload["all_pass"] is True


def test_built_harness_accepts_numeric_output_with_different_line_grouping() -> None:
    reward = CodeTestCaseHarnessReward(toolkit=ToolkitStub("unused"))
    harness = reward.build_harness("print('1\\n2 3\\n4')", ["1\n"], ["1 2\n3 4\n"])

    completed = subprocess.run(
        [sys.executable, "-c", harness],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )

    marker_index = completed.stdout.rfind(_RESULTS_MARKER)
    payload_line = completed.stdout[marker_index + len(_RESULTS_MARKER) :].strip().split("\n", 1)[0]
    payload = json.loads(payload_line)
    assert payload["per_test"][0]["passed"] is True
    assert payload["all_pass"] is True


def test_built_harness_keeps_text_line_structure_significant() -> None:
    reward = CodeTestCaseHarnessReward(toolkit=ToolkitStub("unused"))
    harness = reward.build_harness(
        "print('Problem #1')\nprint('No completed squares can be found.')",
        ["1\n"],
        ["Problem #1\n\nNo completed squares can be found.\n"],
    )

    completed = subprocess.run(
        [sys.executable, "-c", harness],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )

    marker_index = completed.stdout.rfind(_RESULTS_MARKER)
    payload_line = completed.stdout[marker_index + len(_RESULTS_MARKER) :].strip().split("\n", 1)[0]
    payload = json.loads(payload_line)
    assert payload["per_test"][0]["passed"] is False
    assert payload["all_pass"] is False
