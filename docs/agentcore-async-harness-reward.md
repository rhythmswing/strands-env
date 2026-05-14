# AgentCore Async Harness Reward Design

This note defines the design boundary for async AgentCore support and the
minimum hidden-test harness reward. It is intentionally narrower than the
pre-revert harness experiments: the first mainline PR should add reusable
transport and one live reward path without unused files or speculative layers.

## Vocabulary

- **Session**: a long-lived AgentCore Code Interpreter session created by
  `StartCodeInterpreterSession`.
- **Invoke**: one `InvokeCodeInterpreter` call against an existing session.
- **Sample**: one model completion to grade.
- **Test case**: one hidden input/output pair for a sample.

The target execution model is:

```text
start a pool of AgentCore sessions
sample A -> schedule to a vacant session; one invoke containing all tests for sample A
sample B -> schedule to a vacant session; one invoke containing all tests for sample B
sample C -> schedule to a vacant session; one invoke containing all tests for sample C
drain and recycle sessions before expiry
```

This reuses sessions across many samples, but keeps grading, retry, and masking
accounting scoped to one sample at a time.

## Design Principles

1. **Reuse sessions; do not invoke per test.**

   AgentCore sessions are expensive enough to reuse across many samples. Hidden
   tests are numerous enough that one invoke per test case multiplies API
   traffic unnecessarily. A harness should put all selected tests for one
   sample into one sandbox invocation.

2. **Do not batch multiple samples into one invoke in the first PR.**

   Multi-sample invokes couple unrelated rewards. One hung or malformed sample
   can consume the batch budget, one transport failure can invalidate multiple
   rewards, and retry/masking becomes less precise. Per-sample invokes preserve
   a clean training contract while still eliminating per-test API fanout.

3. **Keep the interactive sandbox separate from the grader.**

   `CodeSandboxEnv` is an agent tool environment. It should expose code and
   terminal execution to the model. Hidden-test extraction, harness generation,
   stdout comparison, and reward masking belong in reward code, not in the
   environment.

4. **Put async transport in the generic toolkit.**

   Async AgentCore client creation, async event stream parsing, quota limiters,
   and compatibility with existing sync-style test doubles belong in
   `CodeInterpreterToolkit`. Reward-specific retry policy, masking, and harness
   construction should stay outside the toolkit.

5. **Preserve mainline compatibility.**

   Existing `CodeInterpreterToolkit(client=...)` and `CodeSandboxEnv` call sites
   should continue to work. New async support should be additive, for example
   `create_aio_client(...)` and `CodeInterpreterToolkit(aio_client=...)`.

6. **Separate wrong answers from infrastructure failures.**

   Syntax errors, timeouts inside user code, and wrong stdout are ordinary test
   failures. Quota errors, session death, transport corruption, malformed
   harness output, and sample budget exhaustion are infrastructure-uncertain
   outcomes and should be surfaced as masked rewards so training code can drop
   them instead of treating them as true zeroes.

7. **Isolate test cases inside the sandbox.**

   User code should run in a subprocess per test case inside the AgentCore
   sandbox. This prevents global state leakage between tests and gives a clear
   per-test timeout boundary.

8. **Use one AgentCore session manager.**

   `AgentCorePool` under `src/strands_env/tools/` owns pooled session
   lifecycle. Pooling is transport/session infrastructure, not reward
   semantics. Do not keep both a generic `AgentCorePool` and a reward-only
   session pool in mainline.

## Minimum PR Boundary

The first mainline PR should include only:

- async AgentCore client creation;
- `CodeInterpreterToolkit` support for async clients and async event streams;
- one reusable `AgentCorePool` for pooled session lifecycle;
- backward-compatible sync/test-double behavior;
- one hidden-test harness reward that performs one invoke per sample;
- mocked unit tests for toolkit compatibility and reward parsing/masking;
- this design note.

The first PR should not include:

- multi-sample batching;
- vendored grader files that are not executed;
- duplicate simple and harness reward implementations with overlapping names;
- unused compatibility layers from earlier experiments;
- broad changes to `CodeSandboxEnv` defaults.

`CodeInterpreterToolkit` still supports the simple one-session path. Online
training should route the harness reward through `AgentCorePool` so samples are
scheduled onto vacant sessions, sessions drain before expiry, and session-dead
errors can be retried on a fresh session.

## Deduplication Rules

- There should be one canonical code path for AgentCore invocation parsing in
  `CodeInterpreterToolkit`.
- There should be one canonical pooled session manager:
  `AgentCorePool` under `src/strands_env/tools/`.
- The harness reward should own reward semantics; the toolkit should not know
  about hidden tests.
- Shared helpers should be extracted only when used by at least two live call
  sites in the PR.
- Test fixtures should use small fake clients instead of checked-in sample
  rollout artifacts.
- If a file is included only for future compatibility, leave it out until a live
  caller needs it.

## Follow-Up Criteria

Add richer grading support, such as custom checkers or multiple valid outputs,
only when the dataset contract requires it.

## Real Online Validation

Throughput and reward-quality validation must use real AgentCore sessions and
the same reward path used by training. Use `examples/run_agentcore_harness_real.py`
with a saved training-like candidate trace slice. A representative wave is:

```text
128 prompt batch * 16 rollouts * 2 DAPO oversampling = 4096 reward samples
```

Example:

```bash
PYTHONPATH=src python examples/run_agentcore_harness_real.py \
  --data-path /shared/dev/fengrf/logs/agentcore_harness_real/fulltrace4096_input_20260514T190500Z.jsonl \
  --limit 4096 \
  --sessions 32 \
  --reward-concurrency 64 \
  --session-concurrency 128 \
  --max-tests-per-sample 101 \
  --harness-parallelism 4 \
  --per-test-timeout 10 \
  --invoke-timeout 360 \
  --drain-before-expiry-secs 480 \
  --recycle-before-expiry-secs 120 \
  --read-timeout 480 \
  --save-full-tests \
  --save-full-results \
  --role-arn arn:aws:iam::022992367762:role/CrossAccountAccessRole-prod \
  --output-dir /shared/dev/fengrf/logs/agentcore_harness_real/<run-name>
```

Then analyze reward-risk and throughput artifacts:

```bash
PYTHONPATH=src python examples/analyze_agentcore_harness_artifacts.py \
  /shared/dev/fengrf/logs/agentcore_harness_real/<run-name>
```

The real validation should inspect `summary.json`, `traces.jsonl`,
`throughput_reward_analysis.json`, and targeted reruns for any suspected
false-positive or false-negative rewards.
