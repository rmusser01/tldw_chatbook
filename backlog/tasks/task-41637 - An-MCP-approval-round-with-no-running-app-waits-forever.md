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

**Fix (shared, every round kind).** `InterruptRoundHost.run_round` reads the App once, where it decides whether to mount. The App counts as gone when it is not running and its marshal is its own bound `call_from_thread` (the one that raises "App is not running"). A gone App skips the mount and permission-summary legs, and the round settles as `cancelled` after one poll (about 1 s). For approvals this is the existing fail-closed path: every key is `deny` and listed in `unresolved_keys`, so it is recorded as unanswered, never as a user Deny. Two kinds of host stay live: one without `is_running` (the mobile `HeadlessAppContext`, test doubles), and one whose marshal is routed elsewhere. An App that exits mid-wait is settled by runtime disposal (the visit and shutdown fences), not by this check. ADR-314 has a dated clarification.

**Deviation from the plan.** The first commit (`e8db8db91d`) re-read `is_running` on every poll. The 10-file round subset showed 6 nodes red on head and green on base. An A/B with the guard off and on named the cause: every Console ensure re-points `controller.app` at the wrapped, never-run `TldwCli`, and `test_session_close_pending_journeys` routes that App's `call_from_thread` to the running host. The arm-time, own-marshal check (`b36457ab9d`) makes all 6 pass again. The mid-wait test became "requested after the App exited".

**Evidence** (scratch-profile wrapper, `Tests/Chat/test_console_runtime_lifetime.py`):
- Red first: both new tests failed on the unfixed tree with "the approval round is still waiting" after their 10 s join. A mutation (`app_gone = False`) on the final code fails both the same way. With the fix, both pass in about 1 s each. The whole file passes 39/39, serially in 25-30 s at load ~85.
- Round UI/Chat subset, 10 files at `-n 4`, head 60F vs base 53F. Most failures on both trees are the wrapper's `RecoveryRequired: raw_source_selection_changed` in non-`bootstrap_profile` UI files. The nodes red only on head were rerun alone. On the final head, 4 remain and each also fails on base: `test_background_decision_updates_runtime_attention_until_resolution`, `test_background_approval_toast_escapes_projected_raw_identity`, `test_reprojection_never_renders_an_exhausted_head[exact-skill_script]`, and `test_late_projection_failure_cannot_resurrect_terminal_announcement[approval]`. The last one is flaky with the guard both off and on (1/3 failed in each of 2 alternating runs).
- The other 48 files that drive rounds (`request_mcp_approvals`, `run_round` or the confirm bridges) at `-n 4`: 118 failed / 1850 passed on head at load 65-80, 61 of them the same `RecoveryRequired`. The 118 failing nodes were rerun on base, where 117 fail. The one exception, `test_session_owned_pending_projection_journeys`, passes alone on head, both with the guard on (188 s) and with it off (183 s).

Files: `tldw_chatbook/Chat/console_interrupt_rounds.py`, `Tests/Chat/test_console_runtime_lifetime.py`, `Tests/Architecture/test_module_size_ratchet.py` (6298 -> 6308), `backlog/decisions/314-indefinite-human-approval-waits.md`.
<!-- SECTION:NOTES:END -->
