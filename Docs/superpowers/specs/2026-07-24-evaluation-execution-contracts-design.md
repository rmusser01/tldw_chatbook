# Evaluation Execution Contracts Design

Date: 2026-07-24
Status: Design approved; written specification review pending
ADR:
[ADR-024](../../../backlog/decisions/024-bounded-evaluation-and-tool-worker-execution.md)
Backlog:
[TASK-496](../../../backlog/tasks/task-496%20-%20Enforce-evaluation-execution-and-run-state-contracts.md)

## Summary

Repair the existing evaluation path without introducing a new provider gateway.
The synchronous production chat dispatcher will run off the event loop with a
real per-attempt timeout. Evaluation samples will execute under the configured
concurrency limit, return in deterministic dataset order, and report progress
once as each sample settles. Partial sample failures will retain their stored
results but make the durable run `failed`; caller cancellation will make it
`cancelled` and continue to propagate.

## Verified Problems

- `BaseEvalRunner._call_llm()` awaits synchronous `chat_api_call()`. A
  synchronous mock reproduced `TypeError: object str can't be used in 'await'
  expression`.
- Eval tests commonly patch the dispatcher with `AsyncMock`, hiding the
  production mismatch.
- `max_concurrent_requests`, `request_timeout`, and `retry_attempts` are exposed
  on `EvalRunner` but do not govern the evaluation path.
- Sample evaluation is sequential even when independent samples are available.
- A real two-sample run with one sample failure was stored as `completed` with
  `error_message = NULL`.
- Cancelling an orchestrated run unregisters it in `finally` but leaves the
  database row `running`.
- `Tests/Evals/test_eval_integration.py` patches a
  `QuestionAnswerRunner` import path that does not exist and fails before
  exercising partial-failure recovery.

## Goals

- Match the production provider dispatcher contract.
- Keep the event loop responsive while a synchronous provider call blocks.
- Apply configured timeout and retry settings to every provider attempt.
- Reject invalid execution bounds before provider work starts.
- Bound independent sample execution.
- Preserve returned result order and exact-once progress accounting.
- Retain individual error results while making terminal run state truthful.
- Persist and propagate cancellation.
- Add regressions that fail on the verified production defects.

## Non-Goals

- Rewriting provider handlers or `chat_api_call()` as native async APIs.
- Introducing another provider interface, gateway, queue, or scheduler.
- Force-killing a synchronous provider handler after its thread has started.
- Adding a `completed_with_errors` database status or changing schema.
- Redesigning evaluation metrics, judge prompts, or dataset quality.
- Calibrating LLM-as-judge evaluators; no human-label or judge-calibration
  artifacts were available in this runtime-contract review.
- Repairing the code-execution `RLIMIT_AS`/`RLIMIT_NPROC` portability gap in
  TASK-332.
- Changing Console worker groups, screen ownership, or application state.

## Execution Configuration

The accepted effective settings are:

| Setting | Effective source | Validation |
| --- | --- | --- |
| `max_concurrent_requests` | model execution config, default `10` | positive integer, excluding booleans |
| `request_timeout` | model execution config, default `30.0` seconds | positive finite number, excluding booleans |
| `retry_attempts` | model execution config; task metadata `max_retries` fallback; default `3` | non-negative integer, excluding booleans |
| `retry_delay` | model execution config; task metadata fallback; default `1.0` seconds | non-negative finite number, excluding booleans |

Invalid values fail during runner construction. The basic and specialized
runners receive the same normalized timeout and retry settings so a task
category cannot silently select a different execution policy.

## Provider Boundary

`BaseEvalRunner._call_llm()` keeps the existing message and parameter mapping.
Its invocation changes:

1. call synchronous `chat_api_call()` through `asyncio.to_thread()`;
2. if the returned object is awaitable, await it for test-double and compatible
   adapter support;
3. wrap the complete attempt in `asyncio.wait_for()` using
   `request_timeout`;
4. report the configured timeout in `ExecutionError`, not a hard-coded value.

Basic runners retain their existing `ErrorHandler.with_retry()` call around the
provider operation and receive the normalized retry count and delay.
Specialized runners currently call `LLMInterface.generate()` without that
wrapper, so the interface applies the same `with_retry()` policy around each
generation call. It delegates each attempt to `_call_llm()` and does not add a
second retry layer to basic runners.

The timeout is cooperative at the evaluation boundary. Cancelling the
`to_thread()` await does not terminate a synchronous handler already running.
The late value is ignored, but the handler can continue and a configured retry
can overlap it. This limitation is documented and tested only for prompt return
to the caller; hard process termination is not claimed.

## Bounded Sample Execution

`EvalRunner.run_evaluation()` loads the same sample list and creates one task
per sample. Each task acquires an `asyncio.Semaphore` configured by
`max_concurrent_requests` before calling the selected runner.

The task returns its input index and either:

- the normal `EvalSampleResult`; or
- the existing fatal `EvalSampleResult` representation for an ordinary
  per-sample exception.

Cancellation is not converted into a sample result.

The coordinator consumes tasks as they settle. It stores each result in an
index-aligned slot, updates error/retry counts, and invokes the progress
callback exactly once with a monotonically increasing completed count. The
callback therefore reflects completion order, while the final result list
reflects dataset order.

Progress and persistence callbacks execute in the coordinator, not inside the
sample error-conversion boundary. If a callback raises, the evaluation cancels
and drains outstanding tasks and propagates the callback failure. If the caller
cancels, it likewise cancels and drains outstanding tasks before re-raising
`asyncio.CancelledError`.

The dataset is already loaded into memory, so this tranche does not add a queue
or streaming dataset abstraction merely to avoid creating one task per loaded
sample.

## Durable Run State

The orchestrator continues storing each result from the progress callback and
calculating aggregate metrics after all samples settle.

Before terminal status:

- count results whose `error_info` is non-empty or whose metrics include
  `"error"`;
- compare stored result count with returned result count;
- mark the run `completed` only when both counts are clean;
- otherwise mark it `failed` and store a concise count-based error summary.

Partial failure returns the run identifier normally so callers can inspect
retained results and metrics. A pipeline exception still marks the run failed
and raises the existing enhanced `EvaluationError`.

The orchestrator adds an `except asyncio.CancelledError` branch before its
ordinary exception handler. It attempts to update an existing run to
`cancelled`, never converts cancellation to `EvaluationError`, re-raises the
same cancellation, and unregisters the run in `finally`. A database failure
while recording cancellation is logged but does not replace the cancellation.

## Verification

Focused regression coverage will include:

- a synchronous dispatcher returning a string, executed off the event-loop
  thread;
- an awaitable dispatcher test double;
- timeout and configured retry count/delay;
- invalid execution bounds;
- peak concurrency no greater than the configured limit;
- out-of-order completion with dataset-ordered returned results;
- exact-once progress in settlement order;
- progress/persistence callback failure propagation and outstanding-task
  cleanup;
- partial sample failure persisted as `failed` with retained results;
- caller cancellation persisted as `cancelled`, propagated, and unregistered;
- the corrected `QuestionAnswerRunner` patch path.

Real temporary SQLite is used for orchestrator state assertions. Focused Evals
tests, Ruff, Python compilation, and repository diff checks gate completion.
