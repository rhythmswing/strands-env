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

"""Hidden test-case reward that uses one AgentCore invoke per sample."""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from typing_extensions import override

from strands_env.core.types import Action, RewardFunction, RewardResult, StepResult
from strands_env.tools import AgentCorePool, CodeInterpreterQuotas, CodeInterpreterToolkit
from strands_env.utils.aws import get_client

_RESULTS_MARKER = "__STRANDS_ENV_HARNESS_RESULTS__"

_IMPORT_STRING = (
    "from string import *\n"
    "from re import *\n"
    "from datetime import *\n"
    "from collections import *\n"
    "from heapq import *\n"
    "from bisect import *\n"
    "from copy import *\n"
    "from math import *\n"
    "from random import *\n"
    "from statistics import *\n"
    "from itertools import *\n"
    "from functools import *\n"
    "from operator import *\n"
    "from io import *\n"
    "from sys import *\n"
    "from json import *\n"
    "from builtins import *\n"
    "from typing import *\n"
    "import string\n"
    "import re\n"
    "import datetime\n"
    "import collections\n"
    "import heapq\n"
    "import bisect\n"
    "import copy\n"
    "import math\n"
    "import random\n"
    "import statistics\n"
    "import itertools\n"
    "import functools\n"
    "import operator\n"
    "import io\n"
    "import sys\n"
    "import json\n"
    "sys.setrecursionlimit(50000)\n"
)

_HARNESS_TEMPLATE = r"""
import json
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal, InvalidOperation

_USER_CODE = __IMPORT_STRING__ + __USER_CODE__
_CASES = json.loads(__CASES_JSON__)
_TIMEOUT = __TIMEOUT__
_MAX_WORKERS = __MAX_WORKERS__
_DECIMAL_TOLERANCE = Decimal("1e-6")


def _normalize(value):
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def _stripped_lines(value):
    return [line.strip() for line in _normalize(value).split("\n")]


def _decimal_tokens(line):
    try:
        return [Decimal(tok) for tok in line.split()]
    except (InvalidOperation, ValueError):
        return None


def _compare_outputs(predicted, expected):
    predicted_decimals = _decimal_tokens(_normalize(predicted))
    expected_decimals = _decimal_tokens(_normalize(expected))
    if expected_decimals is not None:
        if predicted_decimals is None or len(predicted_decimals) != len(expected_decimals):
            return False
        return all(
            predicted_token == expected_token
            or abs(predicted_token - expected_token) <= _DECIMAL_TOLERANCE
            for predicted_token, expected_token in zip(predicted_decimals, expected_decimals)
        )

    pred_lines = _stripped_lines(predicted)
    exp_lines = _stripped_lines(expected)
    if len(pred_lines) != len(exp_lines):
        return False
    for predicted_line, expected_line in zip(pred_lines, exp_lines):
        if predicted_line == expected_line:
            continue
        predicted_decimals = _decimal_tokens(predicted_line)
        expected_decimals = _decimal_tokens(expected_line)
        if predicted_decimals is not None and expected_decimals is not None:
            if predicted_decimals == expected_decimals:
                continue
            if len(predicted_decimals) == len(expected_decimals) and all(
                abs(predicted - expected) <= _DECIMAL_TOLERANCE
                for predicted, expected in zip(predicted_decimals, expected_decimals)
            ):
                continue
        return False
    return True


def _run_one(index, test_input, expected_output):
    try:
        with tempfile.TemporaryDirectory() as workdir:
            result = subprocess.run(
                [sys.executable, "-c", _USER_CODE],
                input=test_input,
                capture_output=True,
                text=True,
                timeout=_TIMEOUT,
                cwd=workdir,
            )
        passed = result.returncode == 0 and _compare_outputs(result.stdout, expected_output)
        return {
            "test_case": index,
            "passed": passed,
            "returncode": result.returncode,
            "stdout_head": (result.stdout or "")[:200],
            "stderr_head": "" if passed else (result.stderr or "")[:200],
        }
    except subprocess.TimeoutExpired:
        return {
            "test_case": index,
            "passed": False,
            "returncode": None,
            "stdout_head": "",
            "stderr_head": "TIMEOUT",
        }
    except Exception as exc:
        return {
            "test_case": index,
            "passed": False,
            "returncode": None,
            "stdout_head": "",
            "stderr_head": "%s: %s" % (type(exc).__name__, str(exc)[:200]),
        }


def _main():
    results = {}
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
        future_to_index = {
            executor.submit(_run_one, index, test_input, expected_output): index
            for index, (test_input, expected_output) in enumerate(_CASES)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            results[index] = future.result()

    per_test = [results[index] for index in range(len(_CASES))]
    payload = {
        "per_test": per_test,
        "all_pass": bool(per_test) and all(result["passed"] for result in per_test),
    }
    print("__MARKER__")
    print(json.dumps(payload))


_main()
"""


def _extract_code(content: str, *, last: bool) -> str | None:
    """Extract Python code from the final response."""
    matches = re.findall(r"```python\s*\n(.*?)\n```", content, re.DOTALL | re.IGNORECASE)
    if not matches:
        matches = re.findall(r"```\s*\n(.*?)\n```", content, re.DOTALL)
    if not matches:
        return None
    return matches[-1] if last else matches[0]


@dataclass
class _HarnessOutcome:
    """Parsed harness outcome."""

    passed: int = 0
    total: int = 0
    all_pass: bool = False
    reward_masked: bool = False
    reward_masked_reason: str | None = None
    sample_wall_time_ms: float = 0.0
    last_error_type: str | None = None
    last_error: str | None = None
    test_results_preview: list[dict[str, Any]] = field(default_factory=list)

    def to_info(self) -> dict[str, Any]:
        """Return serializable reward diagnostics."""
        return {
            "passed": self.passed,
            "total": self.total,
            "all_pass": self.all_pass,
            "reward_masked": self.reward_masked,
            "reward_masked_reason": self.reward_masked_reason,
            "sample_wall_time_ms": self.sample_wall_time_ms,
            "last_error_type": self.last_error_type,
            "last_error": self.last_error,
            "test_results_preview": self.test_results_preview,
        }


class CodeTestCaseHarnessReward(RewardFunction):
    """Reward final Python code by running hidden stdin/stdout test cases.

    The reward performs one AgentCore invoke per sample. That invoke runs a
    sandbox-side harness, and the harness runs the selected hidden tests in
    subprocesses so test cases do not share Python global state.
    """

    def __init__(
        self,
        *,
        agent_core_pool: AgentCorePool | None = None,
        toolkit: CodeInterpreterToolkit | None = None,
        client: Any | None = None,
        aio_client: Any | None = None,
        quotas: CodeInterpreterQuotas | None = None,
        session_name: str = "code-test-case-harness-reward",
        extract_last_code_block: bool = True,
        binary_reward: bool = False,
        max_tests_per_sample: int | None = 50,
        harness_parallelism: int = 4,
        per_test_timeout: float = 10.0,
        invoke_timeout: float | None = 360.0,
        test_results_limit: int | None = 5,
    ) -> None:
        """Initialize a `CodeTestCaseHarnessReward` instance."""
        if agent_core_pool is not None and toolkit is not None:
            raise ValueError("Pass agent_core_pool or toolkit, not both")
        if agent_core_pool is not None and (client is not None or aio_client is not None or quotas is not None):
            raise ValueError("Pass agent_core_pool or client/aio_client/quotas, not both")
        if toolkit is not None and (client is not None or aio_client is not None or quotas is not None):
            raise ValueError("Pass toolkit or client/aio_client/quotas, not both")
        if max_tests_per_sample is not None and max_tests_per_sample < 1:
            raise ValueError("max_tests_per_sample must be >= 1 when set")
        if harness_parallelism < 1:
            raise ValueError("harness_parallelism must be >= 1")
        if per_test_timeout <= 0:
            raise ValueError("per_test_timeout must be > 0")
        if invoke_timeout is not None and invoke_timeout <= 0:
            raise ValueError("invoke_timeout must be > 0 when set")
        if test_results_limit is not None and test_results_limit < 1:
            raise ValueError("test_results_limit must be >= 1 when set")

        self.extract_last_code_block = extract_last_code_block
        self.binary_reward = binary_reward
        self.max_tests_per_sample = max_tests_per_sample
        self.harness_parallelism = harness_parallelism
        self.per_test_timeout = per_test_timeout
        self.invoke_timeout = invoke_timeout
        self.test_results_limit = test_results_limit
        self._agent_core_pool = agent_core_pool

        self._owns_toolkit = toolkit is None and agent_core_pool is None
        if toolkit is None and agent_core_pool is None:
            if client is not None:
                resolved_client = client
            elif aio_client is not None:
                resolved_client = None
            else:
                resolved_client = get_client("bedrock-agentcore")
            toolkit = CodeInterpreterToolkit(
                client=resolved_client,
                aio_client=aio_client,
                session_name=session_name,
                quotas=quotas,
            )
        self._toolkit = toolkit

    @override
    async def compute(self, action: Action, step_result: StepResult) -> RewardResult:
        """Compute a hidden-test reward for one model completion."""
        started = time.monotonic()
        content = step_result.observation.final_response
        if content is None:
            return RewardResult(reward=0.0, info={"reason": "no_final_response", "reward_masked": False})

        code = _extract_code(content, last=self.extract_last_code_block)
        if code is None:
            return RewardResult(reward=0.0, info={"reason": "no_code_block_found", "reward_masked": False})

        ground_truth = action.task_context.ground_truth
        if not isinstance(ground_truth, dict):
            return RewardResult(
                reward=0.0,
                info={
                    "reason": "invalid_ground_truth_format",
                    "ground_truth_type": type(ground_truth).__name__,
                    "reward_masked": False,
                },
            )

        test_inputs = list(ground_truth.get("inputs", []))
        test_outputs = list(ground_truth.get("outputs", []))
        if not test_inputs or not test_outputs:
            return RewardResult(
                reward=0.0,
                info={
                    "reason": "no_test_cases",
                    "inputs_count": len(test_inputs),
                    "outputs_count": len(test_outputs),
                    "reward_masked": False,
                },
            )
        if len(test_inputs) != len(test_outputs):
            return RewardResult(reward=0.0, info={"reason": "mismatched_test_lengths", "reward_masked": False})

        if self.max_tests_per_sample is not None and len(test_inputs) > self.max_tests_per_sample:
            test_inputs = test_inputs[: self.max_tests_per_sample]
            test_outputs = test_outputs[: self.max_tests_per_sample]

        harness_src = self.build_harness(code, test_inputs, test_outputs)
        outcome = await self._run_harness(harness_src, total=len(test_inputs), started=started)
        if outcome.reward_masked:
            return RewardResult(
                reward=0.0,
                info={"reason": outcome.reward_masked_reason or "harness_masked", **outcome.to_info()},
            )

        if self.binary_reward:
            reward = 1.0 if outcome.all_pass else 0.0
        else:
            reward = outcome.passed / outcome.total if outcome.total else 0.0

        return RewardResult(reward=reward, info=outcome.to_info())

    def build_harness(self, code: str, inputs: list[str], outputs: list[str]) -> str:
        """Build the sandbox-side Python harness for one sample."""
        cases_json = json.dumps(list(zip(inputs, outputs, strict=True)))
        return (
            _HARNESS_TEMPLATE.replace("__IMPORT_STRING__", repr(_IMPORT_STRING))
            .replace("__USER_CODE__", repr(code))
            .replace("__CASES_JSON__", repr(cases_json))
            .replace("__TIMEOUT__", repr(float(self.per_test_timeout)))
            .replace("__MAX_WORKERS__", repr(int(self.harness_parallelism)))
            .replace("__MARKER__", _RESULTS_MARKER)
        )

    async def cleanup(self) -> None:
        """Clean up the owned Code Interpreter session."""
        if self._owns_toolkit and self._toolkit is not None:
            await self._toolkit.cleanup()

    async def _run_harness(self, harness_src: str, *, total: int, started: float) -> _HarnessOutcome:
        outcome = _HarnessOutcome(total=total)
        try:
            if self.invoke_timeout is None:
                raw = await self._invoke_harness(harness_src)
            else:
                raw = await asyncio.wait_for(
                    self._invoke_harness(harness_src),
                    timeout=self.invoke_timeout,
                )
        except asyncio.TimeoutError as exc:
            return self._masked_outcome(
                outcome,
                started=started,
                reason="harness_invoke_timeout",
                error=exc,
            )
        except Exception as exc:  # noqa: BLE001
            return self._masked_outcome(
                outcome,
                started=started,
                reason="harness_invoke_failed",
                error=exc,
            )

        self._parse_harness_output(raw, outcome)
        outcome.sample_wall_time_ms = (time.monotonic() - started) * 1000.0
        return outcome

    async def _invoke_harness(self, harness_src: str) -> str:
        if self._agent_core_pool is not None:
            return await self._agent_core_pool.invoke(harness_src, language="python")
        if self._toolkit is None:
            raise RuntimeError("CodeTestCaseHarnessReward has no execution backend")
        return await self._toolkit.invoke("executeCode", {"code": harness_src, "language": "python"})

    def _parse_harness_output(self, raw: str, outcome: _HarnessOutcome) -> None:
        idx = raw.rfind(_RESULTS_MARKER)
        if idx < 0:
            outcome.reward_masked = True
            outcome.reward_masked_reason = "malformed_harness_output"
            outcome.last_error_type = "MissingHarnessMarker"
            outcome.last_error = "Harness result marker missing from AgentCore output"
            outcome.test_results_preview = [{"error": "no_marker", "raw_tail": raw[-200:]}]
            return

        line = raw[idx + len(_RESULTS_MARKER) :].strip().split("\n", 1)[0]
        try:
            payload = json.loads(line)
        except Exception as exc:  # noqa: BLE001
            outcome.reward_masked = True
            outcome.reward_masked_reason = "malformed_harness_output"
            outcome.last_error_type = type(exc).__name__
            outcome.last_error = f"{type(exc).__name__}: {exc}"
            outcome.test_results_preview = [{"error": "json_parse_failed", "line_prefix": line[:200]}]
            return

        per_test = payload.get("per_test") if isinstance(payload, dict) else None
        if not isinstance(per_test, list) or len(per_test) != outcome.total:
            outcome.reward_masked = True
            outcome.reward_masked_reason = "malformed_harness_output"
            outcome.last_error_type = "InvalidHarnessPayload"
            outcome.last_error = "Harness payload has invalid per_test results"
            return

        outcome.passed = sum(1 for result in per_test if result.get("passed"))
        outcome.all_pass = outcome.total > 0 and outcome.passed == outcome.total
        outcome.test_results_preview = list(
            per_test if self.test_results_limit is None else per_test[: self.test_results_limit]
        )

    def _masked_outcome(
        self,
        outcome: _HarnessOutcome,
        *,
        started: float,
        reason: str,
        error: BaseException,
    ) -> _HarnessOutcome:
        outcome.reward_masked = True
        outcome.reward_masked_reason = reason
        outcome.last_error_type = type(error).__name__
        outcome.last_error = f"{type(error).__name__}: {error}"
        outcome.sample_wall_time_ms = (time.monotonic() - started) * 1000.0
        return outcome
