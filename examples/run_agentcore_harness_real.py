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

"""Run the AgentCore harness reward on real scored coding data and save traces."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import time
from collections import Counter
from datetime import UTC, datetime
from os import environ
from pathlib import Path
from typing import Any

from botocore.config import Config

from strands_env.core.types import Action, Observation, StepResult, TaskContext, TerminationReason
from strands_env.rewards import CodeTestCaseHarnessReward
from strands_env.tools import AgentCorePool, AgentCorePoolConfig, CodeInterpreterQuotas
from strands_env.utils.aws import get_client


def _utc_slug() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _json_default(value: Any) -> str:
    return str(value)


def _pass_fraction(passed: Any, total: Any) -> float | None:
    if not isinstance(passed, int) or not isinstance(total, int) or total <= 0:
        return None
    return passed / total


def _row_prompt(row: dict[str, Any]) -> str:
    messages = row.get("messages") or []
    if messages and isinstance(messages[0], dict):
        return str(messages[0].get("content", ""))
    return str(row.get("prompt") or "")


def _stored_from_fulltrace(trace: dict[str, Any]) -> dict[str, Any]:
    passed = trace.get("passed")
    total = trace.get("total")
    all_pass = bool(passed == total) if passed is not None and total is not None else trace.get("reward") == 1.0
    generation = trace.get("generation") if isinstance(trace.get("generation"), dict) else {}
    return {
        "all_pass": all_pass,
        "passed": passed,
        "total": total,
        "reward": trace.get("reward"),
        "reason": trace.get("reason"),
        "extract_info": trace.get("extract_info"),
        "generation": {key: value for key, value in generation.items() if key not in {"text"}},
    }


def _extract_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    with Path(args.data_path).open(encoding="utf-8") as f:
        for problem_index, line in enumerate(f):
            if problem_index < args.problem_offset:
                continue
            row = json.loads(line)
            label = row.get("label") or {}
            inputs = list(label.get("inputs") or [])
            outputs = list(label.get("outputs") or [])
            if not inputs or not outputs or len(inputs) != len(outputs):
                continue

            prompt = _row_prompt(row)
            metadata = row.get("metadata") or {}
            row_traces = row.get("traces") or []
            if row_traces:
                for trace_index, trace in enumerate(row_traces):
                    code = trace.get("extracted_code")
                    if not code and not args.include_no_code:
                        continue
                    candidates.append(
                        {
                            "source_format": "fulltrace",
                            "problem_index": problem_index,
                            "problem_id": row.get("id"),
                            "candidate_index": trace.get("sample_index", trace_index),
                            "prompt": prompt,
                            "metadata": metadata,
                            "inputs": inputs,
                            "outputs": outputs,
                            "stored": _stored_from_fulltrace(trace),
                            "code": code,
                        }
                    )
                    if len(candidates) >= args.limit:
                        return candidates
                continue

            for candidate_index, stored in enumerate(row.get("pass_k_reward_info") or []):
                code = stored.get("code_extracted")
                if not code and not args.include_no_code:
                    continue
                candidates.append(
                    {
                        "source_format": "pass_k_reward_info",
                        "problem_index": problem_index,
                        "problem_id": row.get("id"),
                        "candidate_index": candidate_index,
                        "prompt": prompt,
                        "metadata": metadata,
                        "inputs": inputs,
                        "outputs": outputs,
                        "stored": stored,
                        "code": code,
                    }
                )
                if len(candidates) >= args.limit:
                    return candidates
    return candidates


def _make_action(candidate: dict[str, Any], args: argparse.Namespace) -> Action:
    test_inputs, test_outputs = _selected_tests(candidate, args.max_tests_per_sample)
    return Action(
        message=str(candidate["prompt"]),
        task_context=TaskContext(
            ground_truth={
                "inputs": test_inputs,
                "outputs": test_outputs,
                "problem_id": candidate.get("problem_id"),
                "problem_index": candidate.get("problem_index"),
                "candidate_index": candidate.get("candidate_index"),
                "metadata": candidate.get("metadata") or {},
                "original_test_count": min(len(candidate["inputs"]), len(candidate["outputs"])),
            },
            conversation_history=[],
        ),
    )


def _make_step_result(candidate: dict[str, Any]) -> StepResult:
    code = candidate.get("code")
    if code:
        content = f"```python\n{code}\n```"
    else:
        content = str(candidate.get("stored", {}).get("response_sample") or "No final code block.")
    return StepResult(
        observation=Observation(messages=[{"role": "assistant", "content": [{"text": content}]}], metrics={}),
        reward=None,
        termination_reason=TerminationReason.TASK_COMPLETE,
    )


def _selected_tests(candidate: dict[str, Any], max_tests: int | None) -> tuple[list[str], list[str]]:
    inputs = list(candidate["inputs"])
    outputs = list(candidate["outputs"])
    if max_tests is not None:
        inputs = inputs[:max_tests]
        outputs = outputs[:max_tests]
    return inputs, outputs


def _build_trace(
    *,
    candidate: dict[str, Any],
    args: argparse.Namespace,
    reward: float,
    info: dict[str, Any],
    latency_ms: float,
    queue_wait_ms: float,
    execution_latency_ms: float,
) -> dict[str, Any]:
    test_inputs, test_outputs = _selected_tests(candidate, args.max_tests_per_sample)
    stored = candidate.get("stored") or {}
    code = candidate.get("code") or ""
    captured_results = info.get("test_results_preview")
    selected_tests = [
        {"index": index, "input": test_input, "output": test_output}
        for index, (test_input, test_output) in enumerate(zip(test_inputs, test_outputs, strict=True))
    ]
    stored_all_pass = stored.get("all_pass")
    stored_pass_fraction = _pass_fraction(stored.get("passed"), stored.get("total"))
    stored_reward = stored.get("reward", stored_pass_fraction)
    new_all_pass = info.get("all_pass")
    comparable = stored_all_pass is not None and not info.get("reward_masked")
    reward_delta = reward - stored_reward if isinstance(stored_reward, (int, float)) and comparable else None
    return {
        "source_format": candidate.get("source_format"),
        "problem_index": candidate.get("problem_index"),
        "problem_id": candidate.get("problem_id"),
        "candidate_index": candidate.get("candidate_index"),
        "metadata": candidate.get("metadata") or {},
        "prompt_head": str(candidate.get("prompt") or "")[:500],
        "code_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest() if code else None,
        "code_head": code[:1000],
        "original_test_count": min(len(candidate["inputs"]), len(candidate["outputs"])),
        "selected_test_count": len(test_inputs),
        "selected_tests": selected_tests if args.save_full_tests else selected_tests[: args.test_preview_limit],
        "stored": {
            "all_pass": stored_all_pass,
            "passed": stored.get("passed"),
            "total": stored.get("total"),
            "pass_fraction": stored_pass_fraction,
            "reward": stored_reward,
            "reason": stored.get("reason"),
            "format_violation": stored.get("format_violation"),
        },
        "agentcore": {
            "reward": reward,
            "pass_fraction": _pass_fraction(info.get("passed"), info.get("total")),
            "latency_ms": latency_ms,
            "queue_wait_ms": queue_wait_ms,
            "execution_latency_ms": execution_latency_ms,
            "passed": info.get("passed"),
            "total": info.get("total"),
            "all_pass": new_all_pass,
            "reward_masked": info.get("reward_masked"),
            "reward_masked_reason": info.get("reward_masked_reason"),
            "last_error_type": info.get("last_error_type"),
            "last_error": info.get("last_error"),
            "test_results": captured_results if args.save_full_results else None,
            "test_results_preview": None if args.save_full_results else captured_results,
        },
        "comparison": {
            "comparable": comparable,
            "all_pass_match": (bool(stored_all_pass) == bool(new_all_pass)) if comparable else None,
            "reward_delta": reward_delta,
        },
    }


async def _score_one(
    *,
    candidate: dict[str, Any],
    reward_fn: CodeTestCaseHarnessReward,
    semaphore: asyncio.Semaphore,
    args: argparse.Namespace,
) -> dict[str, Any]:
    action = _make_action(candidate, args)
    step_result = _make_step_result(candidate)
    queued_at = time.perf_counter()
    async with semaphore:
        started = time.perf_counter()
        result = await reward_fn.compute(action, step_result)
    finished = time.perf_counter()
    latency_ms = (finished - queued_at) * 1000.0
    queue_wait_ms = (started - queued_at) * 1000.0
    execution_latency_ms = (finished - started) * 1000.0
    return _build_trace(
        candidate=candidate,
        args=args,
        reward=float(result.reward),
        info=result.info,
        latency_ms=latency_ms,
        queue_wait_ms=queue_wait_ms,
        execution_latency_ms=execution_latency_ms,
    )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, default=_json_default, ensure_ascii=False) + "\n")


def _summarize(
    *,
    args: argparse.Namespace,
    traces: list[dict[str, Any]],
    pool: AgentCorePool,
    wall_time_s: float,
    artifact_dir: Path,
) -> dict[str, Any]:
    masked = [trace for trace in traces if trace["agentcore"].get("reward_masked")]
    comparable = [trace for trace in traces if trace["comparison"].get("comparable")]
    mismatches = [trace for trace in comparable if not trace["comparison"].get("all_pass_match")]
    latencies = [float(trace["agentcore"]["latency_ms"]) for trace in traces]
    queue_waits = [float(trace["agentcore"]["queue_wait_ms"]) for trace in traces]
    execution_latencies = [float(trace["agentcore"]["execution_latency_ms"]) for trace in traces]
    return {
        "artifact_dir": str(artifact_dir),
        "config": {
            "data_path": args.data_path,
            "limit": args.limit,
            "problem_offset": args.problem_offset,
            "sessions": args.sessions,
            "reward_concurrency": args.reward_concurrency,
            "max_tests_per_sample": args.max_tests_per_sample,
            "harness_parallelism": args.harness_parallelism,
            "per_test_timeout": args.per_test_timeout,
            "invoke_timeout": args.invoke_timeout,
            "region": args.region,
            "session_timeout_seconds": args.session_timeout_seconds,
        },
        "counts": {
            "total": len(traces),
            "masked": len(masked),
            "comparable": len(comparable),
            "mismatches": len(mismatches),
            "stored_all_pass": sum(1 for trace in traces if trace["stored"].get("all_pass")),
            "agentcore_all_pass": sum(1 for trace in traces if trace["agentcore"].get("all_pass")),
        },
        "masked_reasons": Counter(str(trace["agentcore"].get("reward_masked_reason")) for trace in masked),
        "stored_reasons": Counter(
            str(trace["stored"].get("reason")) for trace in traces if trace["stored"].get("reason")
        ),
        "latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else 0.0,
            "max": max(latencies) if latencies else 0.0,
        },
        "queue_wait_ms": {
            "mean": statistics.fmean(queue_waits) if queue_waits else 0.0,
            "max": max(queue_waits) if queue_waits else 0.0,
        },
        "execution_latency_ms": {
            "mean": statistics.fmean(execution_latencies) if execution_latencies else 0.0,
            "max": max(execution_latencies) if execution_latencies else 0.0,
        },
        "wall_time_s": wall_time_s,
        "samples_per_second": len(traces) / wall_time_s if wall_time_s > 0 else None,
        "pool": pool.stats().as_dict(),
    }


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    artifact_dir = Path(args.output_dir or f"/tmp/agentcore-harness-real-{_utc_slug()}")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    candidates = _extract_candidates(args)
    if not candidates:
        raise RuntimeError("No candidates found")

    quotas = CodeInterpreterQuotas(
        session_concurrency=args.session_concurrency,
        start_tps=args.start_tps,
        invoke_tps=args.invoke_tps,
        stop_tps=args.stop_tps,
    )
    client = get_client(
        "bedrock-agentcore",
        region=args.region,
        profile_name=args.profile_name,
        role_arn=args.role_arn,
        session_name=args.role_session_name,
        config=Config(
            max_pool_connections=args.max_pool_connections,
            connect_timeout=args.connect_timeout,
            read_timeout=args.read_timeout,
            retries={"max_attempts": 3, "mode": "adaptive"},
        ),
    )
    pool = AgentCorePool(
        client=client,
        quotas=quotas,
        config=AgentCorePoolConfig(
            target_size=args.sessions,
            max_in_flight_per_session=1,
            session_timeout_seconds=args.session_timeout_seconds,
            drain_before_expiry_secs=args.drain_before_expiry_secs,
            recycle_before_expiry_secs=args.recycle_before_expiry_secs,
            monitor_interval_secs=args.monitor_interval_secs,
            spawn_concurrency=args.spawn_concurrency,
            session_name_prefix=args.session_name_prefix,
        ),
    )
    reward_fn = CodeTestCaseHarnessReward(
        agent_core_pool=pool,
        binary_reward=args.binary_reward,
        max_tests_per_sample=args.max_tests_per_sample,
        harness_parallelism=args.harness_parallelism,
        per_test_timeout=args.per_test_timeout,
        invoke_timeout=args.invoke_timeout,
        test_results_limit=None if args.save_full_results else args.test_result_preview_limit,
    )
    try:
        await pool.prewarm()
        semaphore = asyncio.Semaphore(args.reward_concurrency)
        started = time.perf_counter()
        traces = await asyncio.gather(
            *[
                _score_one(candidate=candidate, reward_fn=reward_fn, semaphore=semaphore, args=args)
                for candidate in candidates
            ]
        )
        wall_time_s = time.perf_counter() - started
        summary = _summarize(args=args, traces=traces, pool=pool, wall_time_s=wall_time_s, artifact_dir=artifact_dir)

        _write_jsonl(artifact_dir / "traces.jsonl", traces)
        _write_jsonl(
            artifact_dir / "mismatches.jsonl",
            [
                trace
                for trace in traces
                if trace["comparison"].get("comparable") and not trace["comparison"].get("all_pass_match")
            ],
        )
        _write_jsonl(
            artifact_dir / "masked.jsonl", [trace for trace in traces if trace["agentcore"].get("reward_masked")]
        )
        (artifact_dir / "summary.json").write_text(
            json.dumps(summary, default=_json_default, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return summary
    finally:
        await pool.shutdown()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        default="/shared/dev/fengrf/data/competitive_coding/nemotron_full_train_pass8_scored_rl39_20260426.jsonl",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--problem-offset", type=int, default=0)
    parser.add_argument("--include-no-code", action="store_true")
    parser.add_argument("--sessions", type=int, default=4)
    parser.add_argument("--reward-concurrency", type=int, default=8)
    parser.add_argument("--session-concurrency", type=int, default=64)
    parser.add_argument("--start-tps", type=float, default=10.0)
    parser.add_argument("--invoke-tps", type=float, default=30.0)
    parser.add_argument("--stop-tps", type=float, default=10.0)
    parser.add_argument("--max-tests-per-sample", type=int, default=101)
    parser.add_argument("--harness-parallelism", type=int, default=4)
    parser.add_argument("--per-test-timeout", type=float, default=10.0)
    parser.add_argument("--invoke-timeout", type=float, default=360.0)
    parser.add_argument("--binary-reward", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--profile-name", default=None)
    parser.add_argument("--role-arn", default=environ.get("AGENTCORE_ROLE_ARN"))
    parser.add_argument("--role-session-name", default="strands-env-real-harness")
    parser.add_argument("--session-timeout-seconds", type=int, default=900)
    parser.add_argument("--drain-before-expiry-secs", type=float, default=480.0)
    parser.add_argument("--recycle-before-expiry-secs", type=float, default=120.0)
    parser.add_argument("--monitor-interval-secs", type=float, default=2.0)
    parser.add_argument("--spawn-concurrency", type=int, default=4)
    parser.add_argument("--session-name-prefix", default="strands-env-real-harness")
    parser.add_argument("--max-pool-connections", type=int, default=256)
    parser.add_argument("--connect-timeout", type=int, default=60)
    parser.add_argument("--read-timeout", type=int, default=480)
    parser.add_argument("--save-full-tests", action="store_true")
    parser.add_argument("--test-preview-limit", type=int, default=3)
    parser.add_argument("--save-full-results", action="store_true")
    parser.add_argument("--test-result-preview-limit", type=int, default=5)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.limit < 1:
        raise ValueError("limit must be >= 1")
    if args.sessions < 1:
        raise ValueError("sessions must be >= 1")
    if args.reward_concurrency < 1:
        raise ValueError("reward-concurrency must be >= 1")
    if args.max_tests_per_sample < 1:
        raise ValueError("max-tests-per-sample must be >= 1")
    if args.test_result_preview_limit < 1:
        raise ValueError("test-result-preview-limit must be >= 1")
    if args.drain_before_expiry_secs <= args.recycle_before_expiry_secs:
        raise ValueError("drain-before-expiry-secs must be greater than recycle-before-expiry-secs")
    if args.drain_before_expiry_secs >= args.session_timeout_seconds:
        raise ValueError("drain-before-expiry-secs must be less than session-timeout-seconds")
    if args.drain_before_expiry_secs <= args.invoke_timeout:
        raise ValueError("drain-before-expiry-secs must be greater than invoke-timeout")
    if args.read_timeout < args.invoke_timeout:
        raise ValueError("read-timeout must be >= invoke-timeout")


def main() -> int:
    args = _parse_args()
    _validate_args(args)
    summary = asyncio.run(_run(args))
    print(json.dumps(summary, default=_json_default, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
