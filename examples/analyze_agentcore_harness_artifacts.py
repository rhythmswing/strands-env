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

"""Analyze real AgentCore harness run artifacts for throughput and reward risk."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def _json_default(value: Any) -> str:
    return str(value)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _load_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}
    ordered = sorted(values)

    def pick(percentile: float) -> float:
        index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
        return ordered[index]

    return {"p50": pick(0.50), "p90": pick(0.90), "p95": pick(0.95), "p99": pick(0.99)}


def _failed_modes(trace: dict[str, Any]) -> Counter[str]:
    modes: Counter[str] = Counter()
    for result in trace.get("agentcore", {}).get("test_results") or []:
        if result.get("passed"):
            continue
        stderr = str(result.get("stderr_head") or "")
        returncode = result.get("returncode")
        if stderr == "TIMEOUT":
            modes["timeout"] += 1
        elif returncode not in (0, None):
            modes[f"returncode_{returncode}"] += 1
        elif stderr:
            modes["stderr_or_exception"] += 1
        else:
            modes["output_mismatch"] += 1
    return modes


def _passed_nonzero_exit_results(trace: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        result
        for result in trace.get("agentcore", {}).get("test_results") or []
        if result.get("passed") and result.get("returncode") not in (0, None)
    ]


def _risk_record(trace: dict[str, Any]) -> dict[str, Any]:
    modes = _failed_modes(trace)
    agentcore = trace.get("agentcore") or {}
    stored = trace.get("stored") or {}
    comparison = trace.get("comparison") or {}
    return {
        "problem_index": trace.get("problem_index"),
        "problem_id": trace.get("problem_id"),
        "candidate_index": trace.get("candidate_index"),
        "stored": {
            "all_pass": stored.get("all_pass"),
            "passed": stored.get("passed"),
            "total": stored.get("total"),
            "reward": stored.get("reward"),
            "reason": stored.get("reason"),
        },
        "agentcore": {
            "all_pass": agentcore.get("all_pass"),
            "passed": agentcore.get("passed"),
            "total": agentcore.get("total"),
            "reward": agentcore.get("reward"),
            "reward_masked": agentcore.get("reward_masked"),
            "reward_masked_reason": agentcore.get("reward_masked_reason"),
            "execution_latency_ms": agentcore.get("execution_latency_ms"),
            "queue_wait_ms": agentcore.get("queue_wait_ms"),
        },
        "comparison": {
            "all_pass_match": comparison.get("all_pass_match"),
            "reward_delta": comparison.get("reward_delta"),
        },
        "new_failed_test_modes": dict(modes),
        "passed_nonzero_exit_tests": _passed_nonzero_exit_results(trace),
        "code_head": trace.get("code_head"),
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, default=_json_default, ensure_ascii=False) + "\n")


def analyze(run_dir: Path) -> dict[str, Any]:
    traces = _read_jsonl(run_dir / "traces.jsonl")
    summary = _load_summary(run_dir / "summary.json")

    masked = [trace for trace in traces if trace.get("agentcore", {}).get("reward_masked")]
    comparable = [trace for trace in traces if trace.get("comparison", {}).get("comparable")]
    mismatches = [trace for trace in comparable if not trace.get("comparison", {}).get("all_pass_match")]
    false_positive_risks = [
        trace
        for trace in comparable
        if not trace.get("stored", {}).get("all_pass") and trace.get("agentcore", {}).get("all_pass")
    ]
    false_negative_risks = [
        trace
        for trace in comparable
        if trace.get("stored", {}).get("all_pass") and not trace.get("agentcore", {}).get("all_pass")
    ]

    reward_deltas = [
        float(trace["comparison"]["reward_delta"])
        for trace in comparable
        if isinstance(trace.get("comparison", {}).get("reward_delta"), (int, float))
    ]
    execution_latencies = [float(trace.get("agentcore", {}).get("execution_latency_ms") or 0.0) for trace in traces]
    queue_waits = [float(trace.get("agentcore", {}).get("queue_wait_ms") or 0.0) for trace in traces]
    failed_modes = Counter()
    candidates_with_timeout = 0
    passed_nonzero_exit_cases = []
    passed_nonzero_exit_tests = 0
    for trace in traces:
        modes = _failed_modes(trace)
        failed_modes.update(modes)
        if modes.get("timeout"):
            candidates_with_timeout += 1
        passed_nonzero = _passed_nonzero_exit_results(trace)
        if passed_nonzero:
            passed_nonzero_exit_cases.append(_risk_record(trace))
            passed_nonzero_exit_tests += len(passed_nonzero)

    risk_directions = Counter(
        (
            bool(trace.get("stored", {}).get("all_pass")),
            bool(trace.get("agentcore", {}).get("all_pass")),
        )
        for trace in mismatches
    )
    analysis = {
        "run_dir": str(run_dir),
        "summary": summary,
        "counts": {
            "total": len(traces),
            "masked": len(masked),
            "comparable": len(comparable),
            "mismatches": len(mismatches),
            "potential_false_positive_rewards": len(false_positive_risks),
            "potential_false_negative_rewards": len(false_negative_risks),
            "candidates_with_new_timeout_failures": candidates_with_timeout,
            "candidates_with_passed_nonzero_exit_tests": len(passed_nonzero_exit_cases),
            "passed_nonzero_exit_tests": passed_nonzero_exit_tests,
        },
        "mismatch_directions": {
            "stored_fail_new_pass": risk_directions[(False, True)],
            "stored_pass_new_fail": risk_directions[(True, False)],
        },
        "masked_reasons": Counter(str(trace.get("agentcore", {}).get("reward_masked_reason")) for trace in masked),
        "new_failed_test_modes": failed_modes,
        "reward_delta": {
            "mean": statistics.fmean(reward_deltas) if reward_deltas else 0.0,
            "min": min(reward_deltas) if reward_deltas else 0.0,
            "max": max(reward_deltas) if reward_deltas else 0.0,
        },
        "execution_latency_ms": {
            "mean": statistics.fmean(execution_latencies) if execution_latencies else 0.0,
            "max": max(execution_latencies) if execution_latencies else 0.0,
            **_percentiles(execution_latencies),
        },
        "queue_wait_ms": {
            "mean": statistics.fmean(queue_waits) if queue_waits else 0.0,
            "max": max(queue_waits) if queue_waits else 0.0,
            **_percentiles(queue_waits),
        },
        "largest_reward_increases": [
            _risk_record(trace)
            for trace in sorted(
                comparable,
                key=lambda item: (
                    item.get("comparison", {}).get("reward_delta")
                    if isinstance(item.get("comparison", {}).get("reward_delta"), (int, float))
                    else float("-inf")
                ),
                reverse=True,
            )[:20]
        ],
        "largest_reward_decreases": [
            _risk_record(trace)
            for trace in sorted(
                comparable,
                key=lambda item: (
                    item.get("comparison", {}).get("reward_delta")
                    if isinstance(item.get("comparison", {}).get("reward_delta"), (int, float))
                    else float("inf")
                ),
            )[:20]
        ],
    }

    (run_dir / "throughput_reward_analysis.json").write_text(
        json.dumps(analysis, default=_json_default, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_jsonl(run_dir / "potential_false_positive_rewards.jsonl", [_risk_record(t) for t in false_positive_risks])
    _write_jsonl(run_dir / "potential_false_negative_rewards.jsonl", [_risk_record(t) for t in false_negative_risks])
    _write_jsonl(run_dir / "passed_nonzero_exit_cases.jsonl", passed_nonzero_exit_cases)
    return analysis


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    analysis = analyze(args.run_dir)
    print(json.dumps(analysis, default=_json_default, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
