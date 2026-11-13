---
id: TASK-40324
title: "test_stock_console_skill_setup_has_original_owned_lifetime[app_closing] hangs to its 240 s bound now and then"
status: Done
assignee:
- '@claude'
created_date: 2026-11-12 19:36
updated_date: 2026-11-13 04:05
labels:
- follow-up
- tests
- console
- flaky
priority: low
references:
- Tests/Performance/test_console_skill_trust_stock_owned.py
dependencies:
- TASK-38948
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (send stream, TASK-38948's covering runs). Checked by reading the code at the round-13 stack top 8b633d0101, 2026-11-12; line numbers are that tree's.

**Why.** A shutdown-time hang in the stock skill-trust setup costs 4 minutes when it hits and masks real failures in the same run; nothing on CI would notice if it became a real hang on close. TASK-38948 measured it on base and fix alike: base 2 of 16 four-way-parallel runs and 0 of 19 serial; fix 4 of 59. An asyncio task dump showed the script's outer `wait_for` pending with the loop idle.

**Evidence.** `Tests/Performance/test_console_skill_trust_stock_owned.py::test_stock_console_skill_setup_has_original_owned_lifetime[app_closing]`: outer bound `asyncio.wait_for(exercise(), timeout=240)` (`:719`), subprocess `_run(..., timeout=240)` (`:751`), the app_closing branch at `:401-414` and `:471-472`. Last changed by 3e618d7119; on no lane. The same file's `[send_enter]` and `[send_button]` cases are red on the base (TASK-40323 owns them).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The notes say what the pending wait is when the case hangs (a product hang on app close, or a test wait), with the evidence
- [x] #2 The case passes 20 of 20 runs at `-n 4` under load, or a product hang on close is fixed with a test that was red
- [x] #3 When it does hang, it fails within 60 s and names the pending wait
<!-- AC:END -->

## Implementation Plan

1. Re-run the case on base 26b2e677dd; give each wait the script leaves unbounded a 60 s watch that prints every busy task's awaits and every non-idle thread, then exits.
2. Repeat the case four and six at a time until it hangs, and read what it waits on.
3. Fix the wait it names (product or test), with a test that is red first; then 20 runs, four at a time.

## Implementation Notes

**What the pending wait is.** It is a product wait reached in this harness. Three hangs, same report each time:
- The test task is in `run_test`'s teardown: `App._shutdown` -> `App._close_all`, waiting for the ChatScreen's pump to close.
- That pump is inside a handler: `ChatScreen._sync_native_console_chat_ui` -> `prepare_console_presentation_sync` -> `ConsoleReadinessConfigProjection.warm` -> `self._settled.wait()`.
- No thread was working on it.

How it gets there:
- `app_closing` calls `on_shutdown_request()`, which cancels every worker (`_cancel_and_settle_workers`).
- `run_test` installs no task factory. A readiness refresh worker that has not taken its first step is cancelled before `_refresh` starts, so its `finally`, the only place that clears `pending` and sets `_settled`, never runs.
- `warm()` then waits forever, and `pending` would also block every later refresh.
- Under the shipped app's eager task factory, `Worker._start`'s `create_task` runs the read's first step at once, so a cancel lands inside its `try`. That is reasoned from Textual's `Worker._start`, not measured. A probe that installed the eager factory in this fixture stopped earlier, at the fixture's discovery-producer wait (12 of 12), so it says nothing either way.

**Fix (product).** `ConsoleReadinessConfigProjection.run` adds a done-callback to the refresh worker's task, following `raw_cli.py`'s `cancel_unstarted` precedent. A read that is still the projection's `_handoff_read` when its task ends never ran its own `finally`. `_settle_unstarted` clears `pending`, retires its identity handoff, and sets `_settled`. A read that ran is a no-op there. The identity retirement there is advisory, as in `_refresh`'s `finally`, so `_settled` is set even if it raises. Rechecked four-wide afterwards: 4 of 4, load 15.

**Tests.**
- New: `Tests/UI/test_console_readiness_config_projection.py::test_a_refresh_worker_cancelled_before_its_first_step_still_settles`. Red before the fix ("warm() waited on a read that never ran"); green after. It is at the census tail (`MINIMUM_FILES` 2039 after TASK-40322's split). Serial lane time 7.8 s at load 48.
- AC#3 (`test_console_skill_trust_stock_owned.py`): the script's two unbounded waits now get a 60 s watch: the discovery-producer wait, and `run_test`'s teardown (through a small `run_app()` wrapper). When the watch fires it writes the pending report straight to fd 1 (Textual redirects `sys.stdout` inside `run_test`) and exits 3. The parent's 240 s bound stays as the last resort.

**Evidence (four- and six-wide runs, separate processes, since xdist runs a duplicated node id once).**
- Before the fix, `app_closing` hung 7 of 20 (four-wide, load 50-65), then 3 of 6 and 2 of 6. The 7 failed at about 95 s, before the report went to fd 1. With it, 3 of 6 failed at about 100 s, each naming the two frames above, for example: `task message pump ChatScreen(): ... ConsoleReadinessConfigProjection.warm:530 > Event.wait:212` and `stock_skill_hang: App exit (run_test teardown) still pending after 60 s`.
- After the fix: 20 of 20, four-wide, load 49-57, 28-43 s each.

**Covering runs (`-n 4`, load 38-48).** The projection test file and 9 Console projection files: 127 passed, 6 failed. The same 6 fail with base projection code:
- `test_actual_mode_bar_defers_cold_config_and_uses_owned_presentation` (`SimpleNamespace` has no `content`)
- `test_direct_coalesced_control_bar_queues_checked_refresh_without_native_entry`
- `test_cost_settings_only_native_postbody_refusal_and_fresh_retry[owner|source|runtime]`
- `test_actual_close_publishes_real_tabs_before_stock_read_release[same_session_drain]`

The module size ratchet passes.

**Files.** `tldw_chatbook/UI/Console_Modules/console_spend_projection.py`, `Tests/UI/test_console_readiness_config_projection.py`, `Tests/Performance/test_console_skill_trust_stock_owned.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`.
