# TASK-902 Evaluation Execution Contracts Implementation Plan

> Execute this plan with the repository's `executing-plans`,
> `test-driven-development`, `systematic-debugging`, and
> `verification-before-completion` skills. The separate ToolExecutor plan is
> superseded and must not be implemented.

**Goal:** Make evaluation execution match the synchronous production provider
boundary, honor configured bounds, provide reliable sync/async callbacks and
public cancellation, and leave every durable run in a truthful terminal state.

**Architecture:** Keep `chat_api_call()` synchronous and adapt it inside the
existing evaluation runner. `EvalRunner` owns sample concurrency and child-task
cleanup. `EvaluationOrchestrator` owns durable result delivery, run-task
registration, public cancellation, and shutdown. The UI remains an adapter and
does not become a new state owner.

**Tech stack:** Python 3.11+, `asyncio`, SQLite through `EvalsDB`, pytest,
pytest-asyncio, Textual event handlers.

**ADR required:** no

**ADR path:** `backlog/decisions/031-bounded-evaluation-and-tool-worker-execution.md`

**Reason:** ADR-031 already decides the provider, concurrency, callback,
cancellation, and terminal-state boundaries. This plan implements its corrected
contract.

**Design:** `Docs/superpowers/specs/2026-07-24-evaluation-execution-contracts-design.md`

---

## Task 1: Pin configuration and production-dispatcher behavior

**Files**

- Create: `Tests/Evals/test_eval_execution_contracts.py`
- Modify: `tldw_chatbook/Evals/eval_runner.py`

1. Add failing parameterized tests that reject booleans, non-positive
   concurrency/timeouts, negative retries/delays, and non-finite numeric values
   during `EvalRunner` construction.
2. Add a failing test using a synchronous `chat_api_call()` double that records
   its thread ID and returns a string. Assert the event-loop heartbeat advances,
   the provider runs off-loop, and the string is returned without `TypeError`.
3. Add a failing compatibility test whose synchronous dispatcher returns an
   awaitable; assert the awaitable is awaited on the owning loop.
4. Run:

   ```bash
   python -m pytest Tests/Evals/test_eval_execution_contracts.py -q
   ```

   Confirm the new tests fail for the verified reasons.
5. Add small module-local bound validators and normalize an effective copy of
   model execution configuration before constructing the selected runner. Use
   model `retry_attempts`/`retry_delay`, then task metadata fallbacks, then the
   documented defaults.
6. Change `_call_llm()` to call the synchronous dispatcher with
   `asyncio.to_thread()`, await a returned awaitable via
   `inspect.isawaitable()`, and wrap the complete attempt in
   `asyncio.wait_for(request_timeout)`.
7. Re-run the focused file and commit only after it is green.

## Task 2: Apply one timeout and retry policy to basic and specialized runners

**Files**

- Modify: `tldw_chatbook/Evals/eval_runner.py`
- Modify: `Tests/Evals/test_eval_execution_contracts.py`
- Modify: `Tests/Evals/test_eval_runner.py`

1. Add failing tests for exact attempt count and configured retry delay after a
   timeout. Patch `asyncio.sleep` so the test records delays without waiting.
2. Cover a basic runner and one specialized-runner `LLMInterface.generate()`
   path. Assert neither path applies two nested retry loops.
3. Assert timeout errors report the effective configured timeout instead of the
   current hard-coded 30 seconds.
4. Thread normalized retry count, retry delay, and timeout into the existing
   runner-local `ErrorHandler`.
5. Keep basic runners' existing `with_retry()` wrappers. Apply the same wrapper
   once in `LLMInterface.generate()` for specialized runners that call the
   interface directly.
6. Run the new tests plus `Tests/Evals/test_eval_runner.py`. Do not alter the
   production dispatcher or introduce a second provider abstraction.

## Task 3: Bound samples and make callback cleanup exact

**Files**

- Modify: `tldw_chatbook/Evals/eval_runner.py`
- Modify: `Tests/Evals/test_eval_execution_contracts.py`

1. Add red tests using controlled per-sample events:
   - peak active sample calls never exceed `max_concurrent_requests`;
   - completion order differs from dataset order;
   - returned results remain in dataset order;
   - sync and async callbacks each receive
     `(completed, total, EvalSampleResult)` once per delivered result;
   - a callback failure cancels and drains blocked siblings;
   - direct caller cancellation drains siblings and re-raises
     `asyncio.CancelledError`.
2. Capture the loop exception handler and promote unawaited-coroutine/runtime
   warnings so orphan work cannot hide behind a passing assertion.
3. Implement one semaphore and one indexed sample coroutine. Convert ordinary
   per-sample exceptions to the existing fatal `EvalSampleResult`; do not catch
   `asyncio.CancelledError`.
4. Create explicit sample tasks, consume them in settlement order, store into
   index-aligned result slots, and invoke callbacks with a monotonically
   increasing completed count.
5. Add a single callback helper that calls the callback and awaits its return
   value only when `inspect.isawaitable()` is true.
6. In coordinator `finally`, cancel unfinished tasks and drain the complete
   task set with `gather(..., return_exceptions=True)`. Preserve the original
   callback failure or cancellation.
7. Re-run focused tests with `-W error::RuntimeWarning`.

## Task 4: Make durable progress and terminal status truthful

**Files**

- Modify: `tldw_chatbook/Evals/eval_orchestrator.py`
- Modify: `Tests/Evals/test_eval_orchestrator.py`
- Modify: `Tests/Evals/test_eval_integration.py`
- Modify: `Tests/Evals/test_eval_execution_contracts.py`

1. Add real-temporary-SQLite tests proving:
   - storage precedes the user progress callback;
   - sync and async user callbacks are both supported;
   - storage or callback failure marks the run failed and escapes as
     `EvaluationError`;
   - error-bearing sample results remain stored but make the run `failed` with
     a count-based summary;
   - clean results and matching storage count make the run `completed`.
2. Correct the stale `QuestionAnswerRunner` patch target in
   `Tests/Evals/test_eval_integration.py` so the partial-failure test reaches
   the real orchestrator path.
3. Make the orchestrator's internal progress wrapper async. Store the result
   first, then invoke and conditionally await the public three-argument
   callback.
4. Count both non-empty `error_info` and a metrics `"error"` marker before
   selecting terminal status. Compare stored and returned counts and retain
   aggregate metrics on partial failure.
5. Keep per-sample failures inspectable and return the run ID normally; reserve
   raised `EvaluationError` for pipeline/storage/callback failures.

## Task 5: Implement discoverable, public run-ID cancellation

**Files**

- Modify: `tldw_chatbook/Evals/eval_orchestrator.py`
- Modify: `Tests/Evals/test_eval_orchestrator.py`
- Modify: `Tests/Evals/test_eval_execution_contracts.py`

1. Add red tests proving:
   - `run_started_callback(run_id)` fires exactly once after the row is
     `running` and active-task registration exists, before provider work;
   - both sync and async start callbacks work;
   - a start-callback failure marks the run failed and starts no sample;
   - `await cancel_evaluation(run_id)` cancels the real registered task,
     returns after it is drained, stores `cancelled`, and removes both active
     registrations;
   - an unknown or already-settled run returns `False`;
   - direct task cancellation follows the same durable state path and remains
     observable as `CancelledError`.
2. Register `asyncio.current_task()` in `_active_tasks` immediately after the
   durable row is marked `running`. Invoke the start callback through the shared
   sync-or-awaitable helper.
3. Add `except asyncio.CancelledError` before `except Exception`. Attempt the
   `cancelled` status update without replacing the original cancellation if
   SQLite reporting fails.
4. Make `cancel_evaluation()` async. It signals the registered task, drains it
   when it is not the current task, suppresses only the target task's expected
   `CancelledError`, and never pops or writes terminal state itself.
5. Remove active-task and concurrent-run registrations only in the owning
   coroutine's `finally`.

## Task 6: Reconcile live callback consumers and shutdown

**Files**

- Modify: `tldw_chatbook/Evals/ab_testing.py`
- Modify: `tldw_chatbook/Evals/eval_orchestrator.py`
- Modify: `Tests/Evals/test_eval_orchestrator.py`
- Modify: `tldw_chatbook/Evals/README.md`
- Modify: `tldw_chatbook/Evals/DEVELOPER_GUIDE.md`

1. Reconcile the live A/B callback adapter with the three-argument
   sync-or-async contract.
2. Do not restore or test the later-retired
   `tldw_chatbook/Event_Handlers/eval_events.py` gen-2 UI cluster. A future
   full-app consumer must use the durable ID from `run_started_callback` and
   await public cancellation.
3. Add `aclose()` tests: signal every active run before draining, drain all,
   then close SQLite. Make synchronous `close()` raise `RuntimeError` without
   closing when active runs exist; preserve normal inactive cleanup used by
   `quick_eval`.
4. Update public Evals documentation for the callback signatures, run-start
   callback, async cancellation, and shutdown behavior.

## Task 7: Verify and reconcile TASK-902

1. Run focused warning-strict tests:

   ```bash
   python -m pytest \
     Tests/Evals/test_eval_execution_contracts.py \
     Tests/Evals/test_eval_orchestrator.py \
     Tests/Evals/test_eval_runner.py \
     -q -W error::RuntimeWarning
   ```

2. Run the full Evals suite:

   ```bash
   python -m pytest Tests/Evals -q
   ```

3. Run direct orchestrator/runner and live A/B integration tests; do not
   recreate a surrogate application for the retired gen-2 UI.
4. Run Ruff on changed Python files, `python -m compileall` on changed source,
   and `git diff --check`.
5. Review the final diff against every TASK-902 acceptance criterion. Add
   implementation notes with exact test counts and any verified baseline-only
   failures, check all criteria, and only then mark TASK-902 Done.
