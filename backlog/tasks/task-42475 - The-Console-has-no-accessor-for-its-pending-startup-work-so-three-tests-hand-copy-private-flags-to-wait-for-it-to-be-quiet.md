---
id: TASK-42475
title: "The Console has no accessor for its pending startup work, so three tests hand-copy private flags to wait for it to be quiet"
status: Done
assignee:
- '@claude'
created_date: 2026-11-23 09:42
updated_date: 2026-11-24 03:58
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
- `credential read` (the credential status the poll last read is `pending`) and `credential poll pass`. While the poll timer runs, the pass is owed when the poll has recorded no gate yet, when its recorded gate is not committed, or when the readiness revision has moved since it read.
- `config retry <owner>`, one per pending config-rebuild retry;
- `worker <name or group>`, for each unfinished worker whose node is the screen or a widget on it.

Each name is read from the state its producer keeps. The credential poll records the gate it reads in `_console_credential_pre_gate_seen`. That is one attribute write; the poll is otherwise the base code. The accessor reads that record and the readiness revision (a pure counter), so it never reads the credential or starts the `claude-credential-readiness` thread. A credential-status or connection-evidence change therefore shows at the poll's next 0.25 s tick; a revision change shows at once. The two control-bar flags are set lazily in `config_sync.py`, so they are still read with a `False` default. The pin test drives them through their real producers, so a rename there turns it red instead of reading as quiet.

**Tests moved onto it (four).** Each keeps its two-check rule, where it has one, and its assertions.
- `test_console_workspace_context_rail.py::_wait_for_console_startup_settled`: empty on two checks in a row, with a 15 s deadline whose failure names what is pending.
- `test_console_mcp_approval.py::_wait_for_production_console_ready`'s `unsettled()`: the accessor's names, plus its own two focus checks (the focus-restore timer and the initial focus), which are not startup work.
- `test_console_rail_reconciliation.py::_wait_for_inspector_quiet`: the rail checks plus the accessor. It no longer copies the credential pre-gate, so the `shared_connection_evidence` import went.
- `test_console_credential_poll_config.py::test_credential_poll_defers_native_config_contention`: its wait for the restored poll (the snapshot committed, plus a copied list of sync flags and console-sync workers) is now the snapshot and the accessor (review round).

Two waits widened as a result. The workspace and MCP waits now also cover the credential read and poll pass and config retries (TASK-42474's additions). The MCP wait covers all of the screen's workers, not only console-sync ones. One wait narrowed: the workspace and rail waits counted every app worker and now count only the screen's own. A temporary probe logged the unfinished workers outside the screen each time one of the three waits returned, over a whole run of the three files: 79 returns, 74 with none, and 5 MCP waits with the app's own `scheduling` worker (`run`), which the MCP wait never counted.

**Pin test.** `test_console_rail_reconciliation.py::test_console_pending_startup_work_names_each_pending_producer` (admission-sensitive 4 lane). On a mounted production-CSS Console it waits for quiet and checks `()`. Then it drives each producer pending and checks the exact names with no await in between, waiting for quiet after each:
- attach, sync in progress and sync requested, set on the screen;
- `_request_console_control_bar_sync(delayed=True)` and `replay_whole_console_sync`;
- a subscription chat with a recording fake status. The accessor call makes no credential read; one poll tick reads `pending`, after which the accessor names `credential read`;
- a readiness-revision bump, after which the real poll commits the new gate;
- a gate the poll read but could not commit, through the real path: the settings rebuild lock held by another thread, a new revision, and one poll tick. The poll records the gate and fails to take the lock, so the commit lags while the revision matches;
- the never-read state, with nothing recorded or committed. This is set by patching both attributes; the real state exists only between mount and the poll's first tick;
- a probe function decorated with `retry_after_config_rebuild` while `config_read_would_wait()` is true, whose retry then runs;
- a screen worker, while an app worker is not named.

**Red/green.**
- Base `chat_screen.py` (no accessor): red, `AttributeError: 'ChatScreen' object has no attribute 'console_pending_startup_work'`.
- Mutation dropping the replay flag from the accessor: red, `assert 'control-bar replay owed' in ('control-bar sync scheduled',)`.
- Review round: before the accessor read the recorded gate, the subscription case was red with `the accessor read the credential` / `assert [True] == []`.
- Final code: green.

**Soak (AC #4), `-n 4` through `run.sh`.**
- **Final tip, after the review round.** Each pair below ran 20 times in parallel with the other, with 16 bounded busy loops for most of the runs.
  - `test_console_rail_reconciliation.py`, `test_console_right_rail.py`, `test_console_shell_regions.py` and `test_console_credential_poll_config.py`: 20 of 20 runs green, 139 passed each, load 18-52.
  - `test_console_workspace_context_rail.py` and `test_console_mcp_approval.py`: 19 of 20 green, 175 tests each, load 18-54. The one failure was `test_zero_result_workspace_search_stays_editable_and_initializes_without_echo` (run 13), which is red at base (below).
- **First round.** The first two files ran 20 of 20 green (92 passed each) at load 38-143. The workspace and MCP pair ran 10 times at load 52-135. Six runs were green; four had five failures between them, none in a changed wait (no `Console startup work still pending`, `Console still unsettled` or `Inspector still busy`):
  - `test_zero_result_workspace_search_stays_editable_and_initializes_without_echo` (runs 6 and 9) reads the search transition one bare pause after setting the Input's value. It is red at base too: one interleaved run held 15 copies of the base file and 15 of this branch's at load 67-85, and base failed 3 of 15, this branch 1 of 15.
  - `test_captured_raw_shell_session_scope_names_this_console_chat` (run 2) uses a card-only harness, with no Console and no changed code. 60 base repetitions passed at load 78-94.
  - `test_request_mcp_approvals_zero_timeout_keeps_round_armed_for_late_decision` (run 6, load 135): `assert 1.486 >= 1.5`. Fixed in the review round (below), and green in all 20 final-tip runs.
  - `test_approved_definitive_tool_stays_mounted_until_real_terminal[crash]` (run 10) uses its own harness; its `entered.wait(2)` timed out.
- One earlier whole-file run failed `test_overflow_focus_order_and_recovery_stay_within_context_section` once at load about 65. It is a left-rail `_RailHarness` test with no changed code. 40 base repetitions passed, and it passed in all 40 soak runs since.

Other checks:
- the "Run admission-sensitive suites 4" step, serial with the lane flags: 600 passed in 1:01:51 at load 44-110 (the pin test is the one new test);
- `Tests/CI` at `-n 4`: 1823 passed, 2 skipped;
- preflight passed; `ruff check tldw_chatbook --select F` is clean; `Tests/lint/ruff_gate.py` exits 0.

`Tests/Architecture`, serial:
- Final tip: 1492 passed and 5 skipped, at load about 45.
- First round, at load about 100: 4 failed.
  - `test_console_review_selection_controller_boundary.py` pins the ratchet's ChatScreen row (`LIVE_RATCHET_ROW`). It was re-pinned, and the review round folded that re-pin into the first commit so that no commit is red.
  - Three `test_persistent_diagnostic_inventory.py` review-ledger tests hit the 300 s timeout. They passed on rerun (224 s, 167 s and 6 s).

**Review round (2026-11-24).**
- **The accessor read the credential.** On a subscription chat it called `subscription_credential_status(background=True)`, which can start the `claude-credential-readiness` thread outside the poll's own gate. It now reads the gate the poll recorded (see Change). The poll keeps its base code plus that one record line. The pin test asserts that an accessor call makes no credential read; it was red before the change (`assert [True] == []`) and is green after.
- **A fourth hand-copied check.** `test_console_credential_poll_config.py`'s quiet check now reads the accessor.
- **`test_request_mcp_approvals_zero_timeout_keeps_round_armed_for_late_decision`** took `started` after `decider.start()`. A main thread descheduled after the start then measured less than the decider's 1.6 s sleep. A forced 0.2 s main-thread delay after the start reproduces it (`assert 1.4006920829415321 >= 1.5`). `started` is now taken before the start, and the same delay passes. Assertions unchanged.
- **Second review: each poll-pass condition is driven.** The pin test drove only the revision condition. It now also drives the uncommitted gate and the never-read state (see Pin test). Red under mutation:
  - dropping `seen != committed` gives `assert () == ('credential poll pass',)` at the uncommitted-gate check;
  - dropping `seen is None` makes the accessor raise `TypeError: 'NoneType' object is not subscriptable` on the real startup state, already inside `_open_production_inspector`'s wait.

  Both are green again once restored. `test_console_rail_reconciliation.py` then ran 20 times at `-n 4`: 20 of 20 green, 55 passed each, load 13-54, with 16 bounded busy loops.
- **History.** The re-pin of `LIVE_RATCHET_ROW` is folded into the accessor commit, so every commit passes the ratchet tests. Each commit was checked with `test_screen_size_ratchet.py` and the boundary test.
- **Follow-up candidate, red at base.** `Tests/UI/test_console_trace_capture_recovery_flow.py::test_first_trace_factory_failure_has_actionable_uncaptured_recovery[send|refusal]` fails at line 95, `assert len(turn_tasks) == 1` (`assert 0 == 1`): no turn is accepted. It is red at base b31e3e9241 in a detached base worktree. Over four paired runs of both cases at load 15-24, base failed 6 of 8 and this branch 5 of 8. Nothing on this branch touches it.

**Size ratchet.** In `Tests/Architecture/test_screen_size_ratchet.py`, chat_screen went 24544/741 -> 24585/743 (+41 lines, +2 methods), then in the review round -> 24592/742 (+7 lines, -1 method: the pre-gate helper went back into the poll). Each step has a dated comment. `test_console_review_selection_controller_boundary.py`'s `LIVE_RATCHET_ROW` is re-pinned with each step, in the same commit.

**Files.** `tldw_chatbook/UI/Screens/chat_screen.py`, `Tests/Architecture/test_screen_size_ratchet.py`, `Tests/Architecture/test_console_review_selection_controller_boundary.py`, `Tests/UI/test_console_rail_reconciliation.py`, `Tests/UI/test_console_workspace_context_rail.py`, `Tests/UI/test_console_mcp_approval.py`, `Tests/UI/test_console_credential_poll_config.py`.
<!-- SECTION:NOTES:END -->
