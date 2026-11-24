---
id: TASK-42475
title: "The Console has no accessor for its pending startup work, so three tests hand-copy private flags to wait for it to be quiet"
status: Done
assignee:
- '@claude'
created_date: 2026-11-23 09:42
updated_date: 2026-11-24 01:49
labels:
- follow-up
- console
- tests
- hardening
priority: medium
references:
- tldw_chatbook/UI/Screens/chat_screen.py
- Tests/UI/test_console_workspace_context_rail.py
- Tests/UI/test_console_mcp_approval.py
- Tests/UI/test_console_rail_reconciliation.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 22 of the follow-up burn-down (2026-11-23), by TASK-42474's review. Checked on `fu/r22-railrecon` (the round-22 stack a999508ae7 plus TASK-42474); paths start at the repository root.

**Why.** Three tests wait for the Console's startup work to finish before they push or count state, and each keeps its own hand-copied list of private `ChatScreen` attributes:
- `Tests/UI/test_console_workspace_context_rail.py:124`, `_wait_for_console_startup_settled` (TASK-42148): `_console_attach_reconciled`, `_console_sync_in_progress`, `_console_sync_requested`, `_console_control_bar_sync_scheduled`, and the worker list.
- `Tests/UI/test_console_mcp_approval.py`, the `unsettled()` check at about line 1659: the same flags plus `_console_control_bar_replay_whole_sync` and a console-sync worker check.
- `Tests/UI/test_console_rail_reconciliation.py`, `_wait_for_inspector_quiet` (TASK-42474): TASK-42148's flags plus `_console_config_retries` (a pending config-rebuild retry) and a copy of the credential poll's own pre-gate (`subscription_credential_status(background=True)`, `subscription_readiness_revision()`, the shared connection evidence's `version`), compared with `_console_credential_pre_gate` so the wait also covers a first background credential read that has not finished.

The lists already disagree. TASK-42474 found three startup producers (the credential poll's first pass, the first background credential read, and a config-rebuild retry) that TASK-42148's list never saw; under load they re-synced Inspector owners inside a test's observation window. A renamed private attribute switches a check off silently: several are read with `getattr(..., False)`, so a rename reads as "quiet". None of the three sees a startup producer added later.

The product owns this knowledge. A `ChatScreen` method that returns the names of its pending startup work (empty when quiet), with a test that pins each producer to a name, would give the tests one signal that fails loudly when a producer is renamed or added.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Console screen exposes one read-only accessor that names each piece of its pending startup and sync work (attach, native sync running or requested, control-bar sync or replay owed, a credential read or credential-poll pass owed, config-rebuild retries, Console-owned workers) and returns nothing when quiet
- [x] #2 A test pins the accessor: driving each producer into its pending state makes the accessor name it, and a quiet screen returns nothing
- [x] #3 The three tests' waits use the accessor instead of their own attribute lists, and keep their two-consecutive-checks rule and their assertions
- [x] #4 The three test files pass in repeated runs under parallel load (at least 10 runs each, or a soak recorded in the notes)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a public read-only `ChatScreen.console_pending_startup_work()` that names each pending producer from the state it keeps (attach, native sync running or requested, control-bar sync or replay owed, credential read or credential-poll pass owed, config-rebuild retries, this screen's workers) and returns an empty tuple when quiet.
2. Factor the credential poll's pre-gate into one method that the poll and the accessor both read, so the accessor keeps no copy of it.
3. Pin it with a mounted production-Console test in `Tests/UI/test_console_rail_reconciliation.py` that drives each producer pending and checks the name, and checks a quiet screen names nothing; show it red before the accessor exists and under a mutation that drops one producer.
4. Move the three waits (`_wait_for_console_startup_settled`, the MCP approval `unsettled()`, `_wait_for_inspector_quiet`) onto the accessor, keeping their two-check rules and assertions.
5. Soak the three files under parallel load; raise the screen size ratchet by the exact growth.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Change.** `ChatScreen.console_pending_startup_work()` (public and read-only) returns a tuple naming each piece of pending startup and sync work, and `()` when the Console is quiet:
- `attach not reconciled`;
- `sync in progress` and `sync requested` (a native sync);
- `control-bar sync scheduled` and `control-bar replay owed`;
- `credential read` (a background read still `pending`, for a chat on the subscription) and `credential poll pass`. The poll pass is owed while the poll timer runs and its committed pre-gate differs from the one it would read now (TASK-42474's condition).
- `config retry <owner>`, one per pending config-rebuild retry;
- `worker <name or group>`, for each unfinished worker whose node is the screen or a widget on it.

Each name is read from the state its producer keeps. The credential poll's pre-gate moved into `_console_credential_pre_gate_now()`, which the poll and the accessor both call, so no copy of it is kept. The two control-bar flags are set lazily in `config_sync.py`, so they are still read with a `False` default. The pin test drives them through their real producers, so a rename there turns it red instead of reading as quiet.

**Tests moved onto it.** Each keeps its two-check rule and its assertions.
- `test_console_workspace_context_rail.py::_wait_for_console_startup_settled`: empty on two checks in a row, with a 15 s deadline whose failure names what is pending.
- `test_console_mcp_approval.py::_wait_for_production_console_ready`'s `unsettled()`: the accessor's names, plus its own two focus checks (the focus-restore timer and the initial focus), which are not startup work.
- `test_console_rail_reconciliation.py::_wait_for_inspector_quiet`: the rail checks plus the accessor. It no longer copies the credential pre-gate, so the `shared_connection_evidence` import went.

Two waits widened as a result. The workspace and MCP waits now also cover the credential read and poll pass and config retries (TASK-42474's additions). The MCP wait covers all of the screen's workers, not only console-sync ones. One wait narrowed: the workspace and rail waits counted every app worker and now count only the screen's own. A temporary probe logged the unfinished workers outside the screen each time one of the three waits returned, over a whole run of the three files: 79 returns, 74 with none, and 5 MCP waits with the app's own `scheduling` worker (`run`), which the MCP wait never counted.

**Pin test.** `test_console_rail_reconciliation.py::test_console_pending_startup_work_names_each_pending_producer` (admission-sensitive 4 lane). On a mounted production-CSS Console it waits for quiet and checks `()`. Then it drives each producer pending and checks the exact names with no await in between, waiting for quiet after each:
- attach, sync in progress and sync requested, set on the screen;
- `_request_console_control_bar_sync(delayed=True)` and `replay_whole_console_sync`;
- a `pending` credential status on a subscription chat;
- a readiness-revision bump, after which the real poll commits the new gate;
- a probe function decorated with `retry_after_config_rebuild` while `config_read_would_wait()` is true, whose retry then runs;
- a screen worker, while an app worker is not named.

**Red/green.**
- Base `chat_screen.py` (no accessor): red, `AttributeError: 'ChatScreen' object has no attribute 'console_pending_startup_work'`.
- Mutation dropping the replay flag from the accessor: red, `assert 'control-bar replay owed' in ('control-bar sync scheduled',)`.
- Final code: green.

**Soak (AC #4), `-n 4` through `run.sh`.**
- `test_console_rail_reconciliation.py` with `test_console_right_rail.py`: 20 of 20 runs green, 92 passed each, load 38-143.
- `test_console_workspace_context_rail.py` with `test_console_mcp_approval.py`: 10 runs at load 52-135, 175 tests each. Six runs were green. Four runs had five failures between them, none in a changed wait: no `Console startup work still pending`, `Console still unsettled` or `Inspector still busy`.
  - `test_zero_result_workspace_search_stays_editable_and_initializes_without_echo` (runs 6 and 9) reads the search transition one bare pause after setting the Input's value. It is red at base too. One interleaved run held 15 copies of the base file and 15 of this branch's, at load 67-85: base failed 3 of 15, this branch 1 of 15.
  - `test_captured_raw_shell_session_scope_names_this_console_chat` (run 2) uses a card-only harness, with no Console and no changed code. 60 base repetitions passed at load 78-94.
  - `test_request_mcp_approvals_zero_timeout_keeps_round_armed_for_late_decision` (run 6, load 135) failed a controller-only wall-clock bound, `assert 1.486 >= 1.5`.
  - `test_approved_definitive_tool_stays_mounted_until_real_terminal[crash]` (run 10) uses its own harness; its `entered.wait(2)` timed out.
- One earlier whole-file run failed `test_overflow_focus_order_and_recovery_stay_within_context_section` once at load about 65. It is a left-rail `_RailHarness` test with no changed code. 40 base repetitions passed, and it passed in all 20 runs above.

Other checks:
- the "Run admission-sensitive suites 4" step, serial with the lane flags: 600 passed in 1:01:51 at load 44-110 (the pin test is the one new test);
- `Tests/CI` at `-n 4`: 1823 passed, 2 skipped;
- preflight passed; `ruff check tldw_chatbook --select F` is clean; `Tests/lint/ruff_gate.py` exits 0.

`Tests/Architecture`, serial, at load about 100: 1488 passed and 4 failed.
- `test_console_review_selection_controller_boundary.py` pins the screen-size ratchet's ChatScreen row (`LIVE_RATCHET_ROW`). It is re-pinned to 24585/743 with a dated comment, as every earlier raise did, and passes.
- Three `test_persistent_diagnostic_inventory.py::test_task_15103_review_ledger_*` tests hit the 300 s timeout. Rerun at load about 50 they passed (224 s, 167 s and 6 s).

**Size ratchet.** `Tests/Architecture/test_screen_size_ratchet.py`: chat_screen 24544/741 -> 24585/743 (+41 lines, +2 methods), with a dated comment; `test_console_review_selection_controller_boundary.py`'s `LIVE_RATCHET_ROW` is re-pinned with it.

**Files.** `tldw_chatbook/UI/Screens/chat_screen.py`, `Tests/Architecture/test_screen_size_ratchet.py`, `Tests/Architecture/test_console_review_selection_controller_boundary.py`, `Tests/UI/test_console_rail_reconciliation.py`, `Tests/UI/test_console_workspace_context_rail.py`, `Tests/UI/test_console_mcp_approval.py`.
<!-- SECTION:NOTES:END -->
