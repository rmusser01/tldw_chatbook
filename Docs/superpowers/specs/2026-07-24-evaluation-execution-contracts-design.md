# Evaluation Execution Contracts Design

Date: 2026-07-24
Status: Corrected and re-reviewed; implementation plan written
ADR:
[ADR-024](../../../backlog/decisions/024-bounded-evaluation-and-tool-worker-execution.md)
Backlog:
[TASK-496](../../../backlog/tasks/task-496%20-%20Enforce-evaluation-execution-and-run-state-contracts.md)

## Summary

Repair the existing evaluation path without introducing a new provider gateway.
The synchronous production chat dispatcher will run off the event loop with a
real per-attempt timeout. Evaluation samples will execute under the configured
concurrency limit, return in deterministic dataset order, and report progress
once as each durably stored sample settles. Partial sample failures will retain
their stored results but make the durable run `failed`. Both direct task
cancellation and the public run-ID cancellation operation will make the run
`cancelled`, drain owned work, and preserve cancellation as control flow.

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
- The orchestrator's public callback path and the active UI adapter disagree:
  the runner invokes `(completed, total, result)` synchronously, while the UI
  supplies an async `(EvalProgress, result)` callback. The returned coroutine is
  never awaited, so progress, cost, and UI updates can be silently skipped.
- `_active_tasks` is initialized but real runs are never registered in it.
  `cancel_evaluation(run_id)` therefore cannot cancel a production run, removes
  hand-injected test tasks before they settle, and writes state separately from
  the coroutine that owns terminal cleanup.
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
- Give callbacks one typed sync-or-async contract and await callback results.
- Retain individual error results while making terminal run state truthful.
- Expose the durable run ID before provider work and make public cancellation
  target, drain, and clean up the owning task.
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
- Making callbacks concurrent or moving them to worker threads.

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
index-aligned slot and invokes its coordinator callback exactly once with a
monotonically increasing completed count. The callback therefore reflects
completion order, while the final result list reflects dataset order.

All sample tasks are explicitly created and owned by the coordinator. A
`finally` cleanup cancels every unfinished task and drains every created task
with `asyncio.gather(..., return_exceptions=True)`. This cleanup runs for caller
cancellation, callback failure, and any unexpected coordinator failure, so no
sample task continues in the background or emits an un-retrieved exception.
After cleanup, the original exception or `asyncio.CancelledError` propagates.

## Callback Contracts

`EvalRunner.run_evaluation()` and
`EvaluationOrchestrator.run_evaluation()` use one progress callback shape:

```python
progress_callback(completed: int, total: int, result: EvalSampleResult)
```

The callback may return `None` or an awaitable. The coordinator calls it on the
owning event loop, tests the returned value with `inspect.isawaitable()`, and
awaits it when needed. Synchronous callbacks must remain short because they run
on the event loop. The callback is never retried.

The orchestrator's coordinator callback first stores the settled result. Only
after storage succeeds does it invoke the user callback, so a visible progress
update always refers to a durable result. Each durably stored result produces
one user callback in settlement order. A storage failure prevents that result's
user callback, fails the run, and triggers sample-task cleanup. A synchronous
callback exception or an awaited callback failure likewise fails the run and is
not converted into a sample error.

The public UI adapter is changed to accept `(completed, total, result)` and
construct `EvalProgress` internally for widgets that consume that value. The
A/B wrapper retains the same three-argument shape. Tests cover both synchronous
and asynchronous callbacks and assert that no coroutine is left unawaited.
Because callbacks run on the application event loop, the eval event handler
updates loop-owned UI state directly or schedules it with the application's
same-loop mechanism; it does not call `app.call_from_thread()`. Thread-safe
marshalling remains appropriate only for actual thread workers.

`run_evaluation()` also accepts:

```python
run_started_callback(run_id: str)
```

using the same sync-or-awaitable invocation rule. It runs exactly once after
the durable row is marked `running` and the owning `asyncio.Task` is registered,
but before any sample/provider work. This gives UI and service callers the real
run ID needed by the public cancellation API. Failure of this callback follows
the ordinary pipeline-failure path: the run becomes `failed`, active-task
registration is cleaned up, and no sample work starts.

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

After creating the durable run, `run_evaluation()` registers
`asyncio.current_task()` in `_active_tasks[run_id]`. The registration is removed
only by the owning coroutine's `finally` block.

The orchestrator adds an `except asyncio.CancelledError` branch before its
ordinary exception handler. It attempts to update an existing run to
`cancelled`, never converts cancellation to `EvaluationError`, re-raises the
same cancellation, and unregisters the run in `finally`. A database failure
while recording cancellation is logged but does not replace the cancellation.

The public API becomes:

```python
await orchestrator.cancel_evaluation(run_id) -> bool
```

It returns `False` when no active task owns the ID. Otherwise it calls
`task.cancel()`, awaits that task to drain when it is not the current task,
suppresses only the target task's expected `CancelledError`, and returns `True`
after the target has run its terminal-state and unregister cleanup. It does not
pop `_active_tasks` or update the database itself. If invoked by the owning task
it signals cancellation without awaiting itself.

Direct cancellation of the task awaiting `run_evaluation()` follows the same
terminal-state path and still propagates `asyncio.CancelledError` to that
caller. `aclose()` requests cancellation for every registered run, drains all
of them, and closes the database only afterward; the synchronous `close()`
raises `RuntimeError` without closing the database when any active run exists.
UI cancellation awaits the public method and uses the ID delivered by
`run_started_callback`, rather than a separately generated placeholder UUID.
The start callback also binds that ID to the progress adapter and cost tracker,
so every surface refers to the same durable run.

## Verification

Focused regression coverage will include:

- a synchronous dispatcher returning a string, executed off the event-loop
  thread;
- an awaitable dispatcher test double;
- timeout and configured retry count/delay;
- invalid execution bounds;
- peak concurrency no greater than the configured limit;
- out-of-order completion with dataset-ordered returned results;
- exact-once progress in settlement order for both sync and async callbacks;
- real UI callback-adapter signature and awaited cost/progress updates;
- progress/persistence/start-callback failure propagation and
  outstanding-task cleanup without unawaited-coroutine warnings;
- partial sample failure persisted as `failed` with retained results;
- real run ID delivered before sample work;
- public run-ID cancellation targets and drains the registered task, persists
  `cancelled`, and unregisters it;
- direct caller cancellation persisted as `cancelled`, propagated, and
  unregistered;
- asynchronous close drains active runs before closing SQLite;
- the corrected `QuestionAnswerRunner` patch path.

Real temporary SQLite is used for orchestrator state assertions. Focused Evals
tests, Ruff, Python compilation, and repository diff checks gate completion.

## Re-review Record

The corrected contract was checked against `eval_runner.py`,
`eval_orchestrator.py`, `eval_events.py`, `ab_testing.py`, `Evals_DB.py`, and
the existing Evals tests. The second pass resolved:

- the incompatible synchronous three-argument and asynchronous two-argument
  callback shapes;
- unawaited async callback results;
- same-loop UI work incorrectly routed through `call_from_thread`;
- the absence of a real active-task registration and a discoverable durable
  run ID;
- cancellation code that removed ownership and wrote status before the run
  settled;
- sample tasks that were not explicitly drained on callback or coordinator
  failure; and
- database close racing active cancellation cleanup.

No new provider abstraction, scheduler, status value, or application-state
owner is required.
