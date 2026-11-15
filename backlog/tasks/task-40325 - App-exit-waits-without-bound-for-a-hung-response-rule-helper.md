---
id: TASK-40325
title: "App exit waits without bound for a hung response-rule helper"
status: Done
assignee:
- '@claude'
created_date: 2026-11-12 19:36
updated_date: 2026-11-13 02:41
labels:
- follow-up
- console
- response-rules
- shutdown
priority: medium
references:
- tldw_chatbook/Chat/console_runtime.py
- tldw_chatbook/Chat/response_rules/resources.py
- tldw_chatbook/Chat/response_rules/evaluator.py
dependencies:
- TASK-39166
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (load stream, TASK-39166). Checked by reading the code at the round-13 stack top 8b633d0101, 2026-11-12; line numbers are that tree's.

**Why.** If a response-rule helper's HTTP call hangs, quitting the app hangs with it.
- `ConsoleRuntime.dispose()` waits on `_rule_helper_pool.wait_for_cleanup()` with no bound (`Chat/console_runtime.py:6695-6708`); that wait is `while self.unsettled_count: await asyncio.sleep(0.01)` (`Chat/response_rules/resources.py:81-84`).
- A lease settles only in the worker thread's `finally` (`run_sync`, `:214-235`). The evaluator's `asyncio.timeout(min(30, remaining))` (`Chat/response_rules/evaluator.py:239-244`) cancels only the awaiter, because the worker is shielded.
- For most providers the helper's 30 s bound never reaches the HTTP call: `project_chat_handler_kwargs` drops `request_timeout` for every provider whose `PROVIDER_PARAM_MAP` lacks it (`Chat/Chat_Functions.py:1526-1554`). TASK-34356 (In Progress) owns that half (its AC#2); this task owns the unbounded exit wait.
- The only test (`Tests/Chat/test_response_rules_resources.py:231-249`) stubs `chat_api_call_fn` and checks kwargs only.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 App exit with a hung rule helper finishes within a stated bound and records the abandoned helper; a test with a helper that never returns is red before the fix
- [x] #2 A helper that finishes normally still settles its lease before exit
<!-- AC:END -->

## Implementation Plan

1. Bound the exit wait: `RuleHelperPool.wait_for_cleanup` takes an optional timeout; `ConsoleRuntime._dispose_owned` passes a stated constant. On expiry the pool logs the abandoned helpers (count and purposes, no bodies) and teardown continues, so every later owner (store, gateway, DBs) still settles its durable work.
2. Keep a hung helper from holding the process: `asyncio.run` joins the default executor and interpreter exit joins every pool thread, so a helper still in its HTTP call after the bound would keep the process alive until the 120 s exit watchdog hard-exits. Run `run_sync` helper calls on a daemon thread per call instead.
3. Tests, in a file a PR lane already runs (`Tests/Chat/test_response_rules_preparation_custody.py`): a subprocess that disposes a runtime with a helper that never answers (no network) and must exit within the bound, red before the fix; extend the existing normal-finish shutdown test to assert the lease settles before dispose returns.

## Implementation Notes

**Bound: 5 s** (`console_runtime.RULE_HELPER_EXIT_WAIT_SECONDS`) for helpers still in their provider call at exit, after which they are abandoned and logged. Every later owner then settles as before, so no durable write is cut short. The helper's result is already refused at shutdown; only its usage record is lost.

- **Bounded wait.** `RuleHelperPool.wait_for_cleanup(timeout=None)` takes an optional bound. When it expires it logs `response_rule_helpers_abandoned_at_exit count=N purposes=...` (no bodies) and returns. `ConsoleRuntime._dispose_owned` passes the constant. Other callers keep the unbounded wait.
- **Daemon helper threads.** A bounded dispose alone was not enough. `asyncio.run` joins the default executor's threads (up to 300 s) and interpreter exit joins every pool thread, so a helper still in its HTTP call held the process until the exit watchdog hard-exited it at 120 s. Measured with a 4 s helper: `asyncio.run` returned after 4.05 s on `to_thread` and after 0.17 s on a daemon thread. `RuleHelperLease.run_sync` now goes through `run_in_executor` on a small executor that starts one daemon thread per call. The lease still settles in the worker's own `finally`, and the context is still copied. `run_preparation` (local CPU work) and `run_async` (loop tasks) are unchanged.
- **Tests.**
  - New: `Tests/Chat/test_response_rules_preparation_custody.py::test_exit_abandons_a_helper_that_never_answers_after_its_bound`. It runs a subprocess with a `ConsoleRuntime`, a pool and a provider adapter that never answers (no network). It checks that dispose returns within the bound plus 10 s, that the helper is still unsettled, and that the abandoned-helper warning was logged. A watchdog ends the child with `exit_still_waiting` if the whole exit takes more than the bound plus 15 s. Red on base: rc 4, `exit_still_waiting`. Green after: 11 s, imports included. This file is on the Console P0 sandboxed lane (whole file).
  - AC#2: `Tests/Chat/test_console_runtime_shutdown.py::test_shutdown_retains_native_helper_physical_cleanup` now asserts the lease has settled when dispose returns. It goes red with the bound set to 0, at the new assertion (line 70), not the old one.
  - `test_retired_transport_waiting_in_executor_never_calls_adapter` routes every `run_in_executor` call to its blocked executor, so the helper still queues behind the blocker.
- **Runs (`-n 4`, load 61-72).**
  - Six response-rule and shutdown files: 132 passed, 4 failed. With base product code, `test_real_agent_remaining_budget_limits_native_repairs[1-True|2-True]` fail the same way ("original ordinary scratch lease retired"). The other two are load flakes. `test_cancelled_session_hook_drain_still_finalizes_exact_close` times out on `settling.wait(2)` at :1062; it has no rule helper. `test_retired_valid_preparation_never_enters_transport[openai-stop]` is the other. In four runs with base product code, the hook-drain node failed once at :1062, and another run failed 3 of the 9 nodes.
  - Gateway, UI response-rule and preparation files, `-k "auxiliary or rule or helper or exit"`: 93 passed.
  - Builder, deepseek, semantic and first-send preimport files: 92 passed.
  - Module size ratchet passes; `console_runtime.py` is 7,039 lines against its 7,043 row.
- **Diagnostic inventory.** The one added call is `logger.warning("response_rule_helpers_abandoned_at_exit count={} purposes={}", len(purposes), ",".join(purposes))`. It carries a count and the fixed purpose tokens (`learning`/`checking`): no user content, secrets, paths or URLs. Reviewed with `--statements`, then `--write`.
- **Files.** `tldw_chatbook/Chat/response_rules/resources.py`, `tldw_chatbook/Chat/console_runtime.py`, `Docs/security/production-diagnostic-inventory.json`, `Tests/Chat/test_response_rules_preparation_custody.py`, `Tests/Chat/test_console_runtime_shutdown.py`.
