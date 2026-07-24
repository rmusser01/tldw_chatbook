# TASK-497 Tool Worker Contracts Implementation Plan

> Execute this plan with the repository's `executing-plans`,
> `test-driven-development`, `systematic-debugging`, and
> `verification-before-completion` skills. Preserve TASK-492's metadata-only
> history boundary.

**Goal:** Make the configured ToolExecutor limit real, make cancellation
terminal and propagating, clean up every batch child, and close unsubmitted MCP
bridge coroutines.

**Architecture:** `ToolExecutor` remains an async, single-event-loop service.
One semaphore bounds actual `Tool.execute()` calls. Leaf calls contain ordinary
failures and record one metadata-only terminal history item. Batch calls own
explicit child tasks and always drain them. The synchronous MCP bridge retains
coroutine ownership until cross-thread submission succeeds.

**Tech stack:** Python 3.11+, `asyncio`, `concurrent.futures`, pytest,
pytest-asyncio.

**ADR required:** no

**ADR path:** `backlog/decisions/024-bounded-evaluation-and-tool-worker-execution.md`

**Reason:** ADR-024 already decides the tool concurrency, cancellation,
history, batch cleanup, and cross-thread ownership contracts.

**Design:** `Docs/superpowers/specs/2026-07-24-tool-worker-contracts-design.md`

**Dependency:** TASK-492 is Done and defines the terminal-only, payload-free
history record that this task must preserve.

---

## Task 1: Pin construction and concurrency limits

**Files**

- Create: `Tests/Tools/test_tool_executor_workers.py`
- Modify: `tldw_chatbook/Tools/tool_executor.py`

1. Add red parameterized tests rejecting boolean, zero, negative, non-integral
   `max_workers` values and boolean, zero, negative, NaN, or infinite
   `timeout_seconds` values.
2. Add a controlled tool that records active and peak calls. Launch more calls
   than the configured limit and assert peak activity never exceeds it.
3. Add a gate-based test proving a queued call starts after capacity releases
   and its timeout clock does not include queue wait.
4. Run:

   ```bash
   python -m pytest Tests/Tools/test_tool_executor_workers.py -q
   ```

   Confirm failures demonstrate the unused pool and missing validation.
5. Remove `ThreadPoolExecutor`, validate bounds before other construction, and
   store one `asyncio.Semaphore(max_workers)`.
6. Acquire the semaphore only for uncached, valid `Tool.execute()` work. Apply
   `asyncio.wait_for()` after admission and release capacity in every terminal
   path.

## Task 2: Make single-call history and cancellation terminal

**Files**

- Modify: `Tests/Tools/test_tool_executor_workers.py`
- Modify: `Tests/Tools/test_tool_executor_privacy.py`
- Modify: `tldw_chatbook/Tools/tool_executor.py`

1. Add red tests for cancellation while waiting on:
   - cache lookup;
   - the execution semaphore;
   - `Tool.execute()`; and
   - cache write after tool completion.
2. For every begun case, assert `CancelledError` propagates and history contains
   exactly one `cancelled` record with bounded metadata, duration, registered
   argument names only, and no argument/result values.
3. Add timeout and ordinary-error tests proving capacity releases and exactly
   one terminal `timeout`, `parse_error`, or `error` record is appended.
4. Refactor `execute_tool_call()` around one terminal-record helper and an
   outer `except asyncio.CancelledError` covering every awaited stage. Re-raise
   cancellation; do not return an error dictionary for it.
5. Preserve immediate result values, cache hits, cache behavior, defensive
   history copies, the 100-record bound, and TASK-492 persistent metadata.
6. Re-run both focused ToolExecutor files with runtime warnings promoted to
   errors.

## Task 3: Preserve order and drain batch children

**Files**

- Modify: `Tests/Tools/test_tool_executor_workers.py`
- Modify: `tldw_chatbook/Tools/tool_executor.py`

1. Add red tests proving:
   - out-of-order leaf completion still returns request-ordered results;
   - an ordinary leaf failure remains a result dictionary and does not cancel
     siblings;
   - parent batch cancellation cancels queued and executing children;
   - a child `CancelledError` or test-only unexpected `BaseException` cancels
     unfinished siblings;
   - all child tasks are done after propagation and the loop reports no
     un-retrieved task exception.
2. Replace the coroutine list with explicit `asyncio.create_task()` children in
   request order and use normal `asyncio.gather()` for the result path.
3. In `finally`, cancel every unfinished child and drain the complete child set
   with `gather(..., return_exceptions=True)`. Never expose cleanup-gather
   values as tool results.
4. Let the original parent cancellation or unexpected child control flow
   propagate after cleanup.

## Task 4: Remove dead reload lifecycle

**Files**

- Modify: `Tests/Tools/test_tool_executor_workers.py`
- Modify: `tldw_chatbook/Tools/tool_executor.py`

1. Add a red reload test that installs a configured global executor and asserts
   reload replaces it without requiring an `executor`/shutdown attribute.
2. Remove retired pool shutdown and any destructor code that exists only for
   the unused `ThreadPoolExecutor`.
3. Keep reload's current rule that callers must not replace the global while
   they are actively using the retired instance; do not add a manager or
   generation registry.

## Task 5: Close rejected MCP bridge coroutines

**Files**

- Modify: `Tests/Agents/test_mcp_tool_provider.py`
- Modify: `tldw_chatbook/Agents/mcp_tool_provider.py`

1. Promote `RuntimeWarning` to an error for
   `test_invoke_execute_on_closed_loop_returns_error_never_raises` and confirm
   the current failure is the unawaited `execute_hub_tool` coroutine.
2. Add a focused submission-failure test that distinguishes pre-transfer from
   post-transfer ownership. The pre-transfer coroutine must be closed exactly
   once; a successfully submitted coroutine must not be closed by the worker
   thread.
3. Construct `execute_hub_tool()` into a named local. If
   `run_coroutine_threadsafe()` raises before returning a future, close the
   local coroutine before returning the existing contained `ToolResult`.
4. Preserve the existing bounded `future.result()` wait, best-effort future
   cancellation, audit discrimination, never-raise API, and error truncation.
5. Run the full MCP provider file with runtime warnings treated as errors.

## Task 6: Verify and reconcile TASK-497

1. Run focused warning-strict tests:

   ```bash
   python -m pytest \
     Tests/Tools/test_tool_executor_workers.py \
     Tests/Tools/test_tool_executor_privacy.py \
     Tests/Agents/test_mcp_tool_provider.py \
     -q -W error::RuntimeWarning
   ```

2. Run all Tool tests and the relevant agent/chat integrations:

   ```bash
   python -m pytest \
     Tests/Tools \
     Tests/Agents/test_mcp_tool_provider.py \
     Tests/Chat/test_console_agent_bridge.py \
     Tests/Chat/test_console_agent_swap.py \
     -q
   ```

3. Run Ruff on changed Python, `python -m compileall` on changed source, and
   `git diff --check`.
4. Re-run the TASK-492 ToolExecutor privacy tests after every lifecycle change.
5. Review the final diff against every TASK-497 acceptance criterion. Add
   implementation notes with exact warning-strict and integration test counts,
   check all criteria, and only then mark TASK-497 Done.
