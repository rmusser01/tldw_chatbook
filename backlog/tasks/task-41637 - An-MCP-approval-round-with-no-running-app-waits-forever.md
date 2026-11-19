---
id: TASK-41637
title: "An MCP approval round with no running app waits forever"
status: Done
assignee: []
created_date: 2026-11-17 19:05
labels:
- follow-up
- mcp
- agents
priority: medium
references:
- Tests/Chat/test_workflow_planning_native.py
dependencies:
- TASK-41314
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Checked on dev 97f8c58c0c; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** While retargeting the planning nodes, TASK-41314 saw that an MCP tool approval requested when no app is running, or once the app has gone, never settles. The round waits with no deadline and no refusal, so the run hangs instead of failing closed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An approval round with no running app settles as refused (fail closed) promptly
- [x] #2 A test requests an approval with the app absent and asserts the refusal within a bounded time
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR (fail-closed on an unanswerable round is the existing cancellation contract); ADR-314 gets a dated clarification, since it says rounds stay armed until answered or stopped.

1. Red first: two tests in `Tests/Chat/test_console_runtime_lifetime.py` request an approval with (a) a constructed, never-started Textual App and (b) an App that exits while the round waits; each asserts an all-`deny`, all-unresolved verdict within 10 s.
2. Fix once in the shared `InterruptRoundHost.run_round` (every round kind waits there): an app object that reports `is_running` False cannot show a card or deliver an answer, so skip the mount/summary legs and settle the round as `cancelled` (fail closed, audited as unresolved). Duck-typed hosts without `is_running` (mobile `HeadlessAppContext`, test doubles) stay live.
3. Mutation check, then run the files that drive approval rounds at `-n 4`.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Cause.** `request_mcp_approvals` refused only when `controller.app is None`. A constructed but never-run `TldwCli` (the private-App planning child), or an App that has exited, is not `None`. The round armed, took the detached/announce path (no marshal), and then polled with no deadline (ADR-314 default). The only call that noticed the dead App was the attention hook (`on_pending_rounds_changed` -> `call_from_thread`: "App is not running"), which swallows the error by design.

**Fix (shared, every round kind).** `InterruptRoundHost.run_round` reads the App once, where it decides whether to mount. The round is unanswerable when the App is not running, its marshal is its own bound `call_from_thread` (the one that raises "App is not running"), and no Console view can show the card (`_approval_view_is_detached`). An unanswerable round skips the mount and permission-summary legs and settles as `cancelled` after one poll (about 1 s). For approvals this is the existing fail-closed path: every key is `deny` and listed in `unresolved_keys`, so it is recorded as unanswered, never as a user Deny. Three kinds of host stay live:
- one without `is_running` (the mobile `HeadlessAppContext`, test doubles);
- one whose marshal is routed to another App;
- any host while a view can still show the card.

The read is plain attribute access with an `AttributeError` fallback, so the app string-lookup census stays at its pin. An App that exits mid-wait is settled by runtime disposal (the visit and shutdown fences), not by this check. ADR-314 has a dated clarification.

**Deviation from the plan (three rounds).**
1. `e8db8db91d` re-read `is_running` on every poll. The 10-file round subset turned up 6 nodes red on head and green on base. Console harnesses re-point `controller.app` at the wrapped, never-run `TldwCli` on every Console ensure, and `test_session_close_pending_journeys` routes that App's `call_from_thread` to the running host. An A/B (guard off and on) on that node named the cause.
2. `b36457ab9d` read liveness once, at arm time, through the App's own marshal. The mid-wait test became "requested after the App exited".
3. `7b8ad72c8b` came from the serial `Tests/Architecture` run, which caught two new `getattr(app, "...")` literals (`test_app_string_lookups_match_their_pin[Chat]`, 175 vs 173). The rules forbid raising that pin, so the read became attribute access. Instrumented harness runs then showed `controller.app` re-pointed at the never-run `TldwCli` beside an answerable view, so the round also requires a detached view.

**Evidence** (scratch-profile wrapper, `Tests/Chat/test_console_runtime_lifetime.py`):
- Red first: both new tests failed on the unfixed tree with "the approval round is still waiting" after their 10 s join. A mutation (`app_gone = False`) on the final code fails both the same way, and the census and size ratchet pass (264 passed). With the fix, both pass in about 1 s each. The whole file passes 39/39, serially in 25-30 s at load ~85.
- Round UI/Chat subset, 10 files at `-n 4`, head 60F vs base 53F. Most failures on both trees are the wrapper's `RecoveryRequired: raw_source_selection_changed` in non-`bootstrap_profile` UI files. The nodes red only on head were rerun alone. On the final head, 4 remain and each also fails on base: `test_background_decision_updates_runtime_attention_until_resolution`, `test_background_approval_toast_escapes_projected_raw_identity`, `test_reprojection_never_renders_an_exhausted_head[exact-skill_script]`, and `test_late_projection_failure_cannot_resurrect_terminal_announcement[approval]`. The last one is flaky with the guard both off and on (1/3 failed in each of 2 alternating runs).
- The 48-file run below was on `b36457ab9d`. The final guard fails closed in a subset of that version's cases, so it can add no new failure there.
- The other 48 files that drive rounds (`request_mcp_approvals`, `run_round` or the confirm bridges) at `-n 4`: 118 failed / 1850 passed on head at load 65-80, 61 of them the same `RecoveryRequired`. The 118 failing nodes were rerun on base, where 117 fail. The one exception, `test_session_owned_pending_projection_journeys`, passes alone on head, both with the guard on (188 s) and with it off (183 s).

- Six harness nodes (the parallel-runs mount/park nodes, tab-close pending, pending projection), 3 runs at `-n 3` on the final head, with the guard's inputs logged whenever the App looked gone. Every such arm (3) saw the re-pointed `TldwCli` and kept the round live, because a view was attached. Run 1 had 3 failures, which are exactly those arms: the mount marshal on the non-running App raises, a pre-existing harness race. Runs 2 and 3 passed 6/6. Base fails `test_background_skill_install_confirm_parks_badges_toasts_and_mounts_on_visit` and `test_new_session_clears_a_mounted_card_from_the_session_being_left` in 2 of 3 runs too.
- `Tests/Architecture`, serial, on `55285ede68` (before the census fix): 4 failed / 1475 passed in 1606 s. Three of the failures also fail on base with identical messages: `test_reviewed_diagnostic_changes_are_metadata_only` and the two `test_timer_path_static_update_inventory` nodes. The fourth was the app string-lookup census, fixed in `7b8ad72c8b`. On the final head, the census and size ratchet pass (264 passed). The timer, code-identity, layer and diagnostic files show 3 failed / 120 passed, with failure messages identical to base.

Files: `tldw_chatbook/Chat/console_interrupt_rounds.py`, `Tests/Chat/test_console_runtime_lifetime.py`, `Tests/Architecture/test_module_size_ratchet.py` (6298 -> 6307), `backlog/decisions/314-indefinite-human-approval-waits.md`.

Review round (2026-11-19): the guard is now `not is_running` AND no answerable view. Mutation showed the `call_from_thread.__self__ is app` marshal condition was not needed once the view condition exists (45 passed in the lifetime + tab-close files without it), so it was dropped; ADR-314 clarification rewritten to say plainly the view condition accommodates harnesses that re-point `controller.app` at a never-run TldwCli beside a live view. Added tests with a real ConsoleRuntime: never-run App + no view fails closed; never-run App + answerable view stays pending. Mutations: dropping the view condition fails the pending test; dropping the guard fails the three refusal tests.
<!-- SECTION:NOTES:END -->
