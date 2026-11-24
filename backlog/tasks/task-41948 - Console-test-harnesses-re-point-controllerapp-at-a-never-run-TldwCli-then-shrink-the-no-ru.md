---
id: TASK-41948
title: "Console test harnesses re-point `controller.app` at a never-run TldwCli; then shrink the no-running-app guard"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-24 04:09'
created_date: '2026-11-19 07:30'
labels:
- follow-up
- tests
- console
priority: low
references:
- Tests/UI
- tldw_chatbook/Chat/console_interrupt_rounds.py
- backlog/decisions/314-indefinite-human-approval-waits.md
dependencies:
- TASK-41637
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41630..41655 follow-up burn-down (2026-11-18/19). Checked on dev d0486978dd unless stated; paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** TASK-41637's guard fails a round closed only when the App is not running AND no Console view can answer, because harnesses (e.g. `ProductionConsoleHarness`) point `controller.app` at a never-run `TldwCli` next to a live view (about 6 nodes depend on it). If the harnesses pointed at the running host, the guard could be just `not app.is_running` (with the AttributeError fallback).

2026-11-23 (round-22 filing): TASK-42465 proposes the other fix for the same guard: judge the arm-time liveness check by the attached view's own App (`view.app.is_running`) instead of repointing the harnesses. Its AC#1 asks the owner to choose between the two; if this task's harness fix is chosen, TASK-42465 closes as covered here.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The harnesses point `controller.app` at the running host: the factory `TldwCli` that every Console ensure hands the controller as `controller.app` is marked running and routes its `call_from_thread` through the harness host while the host runs
- [x] #2 The guard drops the view condition; ADR-314's clarification is updated; the round and harness tests pass
<!-- AC:END -->

AC #1 refined 2026-11-23, while implementing: `controller.app` cannot literally become the bare host App. Every Console ensure re-points it at `ChatScreen.app_instance` (`attach_review_presenter`), and the controller reads that object's `TldwCli` services (`unified_mcp_service`, `app_config`, `console_runtime`, `chachanotes_db`, ...) while the tests read its toast sink. So the harness keeps that object, marks it running and routes its marshal through the host; the running mark is what the guard reads.

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Owner decision 2026-11-23: "Fix the harnesses (Recommended)" -- this task's approach over TASK-42465's.

1. Census, by evidence rather than grep alone: a temporary probe in `run_round` logs every arm whose App reports `is_running` False, with the test id and whether the view condition kept it live. Run every test file that arms rounds or hosts a Console with a controller (about 330 files) at `-n 4`. The rows that stayed live only because of the view condition are the harness dependants. Revert the probe.
2. Point each Console harness's controller app at the running host: one shared helper binds the wrapped, never-run `TldwCli` (the object every Console ensure hands the controller as `controller.app`) to the host App that runs the view, so its `is_running` and `call_from_thread` are the host's, and unbinds it when the host shuts down. Call it from every harness that mounts `ChatScreen` for a wrapped app; drop the per-test `app.call_from_thread = host.call_from_thread` lines it replaces.
3. Guard: `app_gone = not app.is_running` with the `AttributeError` fallback; drop the view condition and update its comment.
4. Tests, red first: the lifetime test that pins "never-run App + answerable view stays pending" becomes "never-run App fails closed even while a view can answer" (the ADR change requires that edit); a harness journey arms a round while a modal covers the view and shows it keeps waiting and is answered after the modal closes (red on base: it fails closed).
5. Decide `_await_answerable_view` and the card wait in `test_console_session_tab_close.py` with the TASK-42481/42424 forced-delay reproductions run without them on the fixed tree.
6. ADR-314's TASK-41637 clarification, the guard comment, and the lessons entries that describe the old rule.
7. Soak the session-close, interrupt-round and harness-using files 20x under load; Tests/CI, Tests/Architecture serially, preflight, ruff F, ruff gate, the touched lane steps.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Owner decision 2026-11-23: "Fix the harnesses (Recommended)", over TASK-42465's alternative.

**Approach.**
- Harness. `Tests/UI/app_factory.py::run_on_host(app, host)` sets the factory `TldwCli`'s `call_from_thread` to the host's and marks it running. A wrapper on the host's `_shutdown` clears the running mark when that shutdown starts and restores the app's own marshal only after it returns. The shipped app behaves the same: `is_running` is False from the start of `_shutdown`, but its marshal lands until `_shutdown` returns. Every harness that mounts `ChatScreen` for a factory app calls it from `on_mount`, so `controller.app` reports running and marshals through the host. That is 25 harness classes in 21 files, among them the shared gate-1 `ConsoleHarness`, which `ProductionConsoleHarness` and the other subclasses inherit, and `DeferredInitialConsole`, which prevents the base mount. The session-close tests' per-test `app.call_from_thread = host.call_from_thread` lines are gone. Why AC #1 is worded this way is noted under the criteria.
- Guard. `InterruptRoundHost.run_round` now uses `app_gone = not app.is_running`, still with the `AttributeError` fallback; the view condition is dropped, and the guard comment and ADR-314's TASK-41637 clarification state the new rule. `console_interrupt_rounds.py` drops from 6304 to 6300 lines; its pin (6307) is unchanged.
- TASK-41637's tests. One was edited, because the ADR change requires it: `test_never_run_app_round_stays_pending_while_a_view_can_answer` pinned the dropped rule. It is now `test_never_run_app_round_is_refused_even_while_a_view_can_answer`. The rest pass unedited: the never-run App, exited App and no-view refusals, the running App that waits under a covering screen, and the `_ThreadApp` host without `is_running`, which still waits.
- Waits in `test_console_session_tab_close.py`, decided with the evidence below.
  - `_await_answerable_view` is removed. It waited for the check's own former input, which the check no longer reads.
  - The viewed-card wait is removed. It was there only to synchronise with the check.
  - Kept: the pre-Close `assert not sibling.done()`, and the `_parked_decision_notice` waits. Those assert that the user is told the closing tab's decision waits, and that it is waiting before Close. Their comments no longer cite the old rule.
- New journey. `_verify_question_armed_under_a_modal_waits_and_is_answered_after_it`, in `test_session_close_pending_journeys`, arms the viewed tab's question while the Close dialog covers Console. After 2.5 polls the round is still live. It then presses Stay, answers the card, and gets `{"answered": True, ...}`.

**Red/green** (scratch probe on PYTHONPATH, never in the repo; logs in `$FU/h24-logs/`).
- Lifetime test, old guard: red, "the approval round is still waiting". New guard: the file passes, 42 passed.
- Old world (base guard and base harness, waits removed, run serially):
  - New journey: red 3/3, `{'answered': False, 'reason': 'cancelled'}`.
  - TASK-42481's hold (first attach reconciliation of the chat-create pass held 0.6 s): red, "closing the background tab answered the viewed tab".
  - TASK-42424's hold (the viewed question's arm-time read held until the dialog covered the view): the round read `app_gone=True`. The test outran its 1 s poll in that run, so the new journey is the deterministic form of this case.
- New tree, the same three: green. The probe logged `app_running=True`, and `app_gone=False` for every arm, including one before the reconciliation and one under the dialog.
- New guard with the gate-1 harness unbound: the pending journeys fail, and the census logged a never-run `TldwCli` arm beside an answerable view. So the binding is required, and the probe was live in private-profile children.

**Census of what the guard change touches.** A read-only probe logged every round arm whose App reported `is_running` False. It ran over 414 files at `-n 4`: every file that runs a bound harness, arms rounds directly, or hosts a real App in a round-related file. The run took 6 h 11 min at load 15-126. It found 4 arms, all in TASK-41637's lifetime tests. The only one beside an answerable view is the rewritten test. No harness test arms a round on a not-running App any more.

**Checks.**
- 414-file run: 10368 passed; 1873 nodes failed or errored, mostly the local `RecoveryRequired` trap in non-private UI tests. All 1873 were rerun on base with the same flags.
  - 1840 also fail on base.
  - 26 of the other 33 passed when rerun on this tree.
  - 5 of the remaining 7 fail on base too.
  - The last 2 are flaky on both trees. `test_request_skill_script_confirm_parks_for_a_non_active_session` failed 1 of 8 serial runs on base and 1 of 8 here, with the same message: it reads the notice right after the round registers. Its fake app has no `is_running`, so the guard does not touch it. `test_library_source_snapshot_times_out_to_stable_error` then passed 5 of 5 here.
- Six harness files (tab-close, parallel-runs, lifetime, initial-draft, hook-review freeze, tab-switch) at `-n 4`: 33 failed on this tree and 35 on base. Every failure here also fails at base: they are local `RecoveryRequired` failures in non-private tests. The two that fail only on base are harness round nodes (`test_background_skill_script_confirm_parks_badges_toasts_and_mounts_on_visit`, `test_mounted_round_survives_switch_away_and_switch_back`).
- Soak: the session-close, lifetime, interrupt-rounds, parallel-runs and initial-draft files 20 times at `-n 4`, beside the 414-file run, at load 28-83. All 20 runs passed 93 of 93, in 167-486 s. The 24 parallel-runs nodes that fail locally with `RecoveryRequired` at base and head were deselected.
- The pending journey, 5 times serially with the UI lane's flags (`-p no:xdist --timeout=180`): 5/5, with calls of 70-94 s at load 40-57.
- `Tests/Architecture`, serial: 1492 passed, 5 skipped. `Tests/CI` at `-n 4`: 1823 passed, 2 skipped. `preflight.sh`: all checks passed.
- `ruff --select F tldw_chatbook`: clean. `Tests/lint/ruff_gate.py`: rc 0. The touched test files have 4 F401s, the same 4 as at base.
- Lane steps, serially: admission-sensitive suites 4, 599 passed; the Console P0 regression tests, 500 passed; admission-sensitive suites 2, 1512 passed.
- There are no new census entries and no User Guide change: the shipped app's `controller.app` was always running.

**Files.** `Tests/UI/app_factory.py`, `tldw_chatbook/Chat/console_interrupt_rounds.py`, `Tests/Chat/test_console_runtime_lifetime.py`, `Tests/UI/test_console_session_tab_close.py`, the 21 harness files, `backlog/decisions/314-indefinite-human-approval-waits.md`, and `backlog/docs/lessons-testing-evidence.md`. The lessons file has a new entry, and the TASK-42424 and TASK-42481 entries no longer give the retired waiting advice.
**Review round (2026-11-24, six Minors).**
1. Shutdown order. `run_on_host` cleared the factory app's marshal before awaiting the host's `_shutdown`. The shipped app's marshal keeps landing until `_shutdown` returns, and the Console runtime drain (`app_lifecycle.py` `_shutdown_console_runtime`) relies on that.
   - Fix: the wrapper now clears the running mark, awaits the host's shutdown, and restores the app's own marshal in `finally`.
   - New test `Tests/UI/test_app_factory_cleanup.py::test_run_on_host_keeps_the_marshal_until_the_host_shutdown_returns`: a thread worker that the host drains in its `_shutdown` lands its result through `app.call_from_thread`, and sees `is_running` False. Red with the old order ("the drained worker's result never landed"), green after.
2. The docstring of `test_running_app_round_stays_pending_while_no_view_can_answer` no longer describes the dropped view rule.
3. Leftover re-points. `test_console_parallel_runs.py` (7 sites) and `Tests/integration/test_console_chat_fork_flow.py` no longer set `controller.app = host`, and their false comments are gone. The `notify` capture moved to the factory app, which both the controller's notices and the screen's park toast call.
   - `test_park_toast_survives_a_viewed_run_completion_re_invocation` then saw two "needs approval" toasts. One is the round's own app-owned announcement ("... needs approval to use a tool ..."). The other comes from the legacy screen seam that the test re-invokes ("Agent in ... needs approval."): that seam dedupes only rounds it toasted itself. The second toast had always gone to an object nobody recorded. The test now counts the round's own announcement, the only toast it ever recorded, and says why in a comment. The seam gap is a proposed follow-up; no production path re-invokes the seam for retained approval rounds.
   - Proof. All seven formerly re-pointed parallel-runs tests are already private-profile, and all pass at `-n 4`. The file's other 22 failures are the local `RecoveryRequired` trap and also fail on base. On base the file also failed `test_mounted_round_survives_switch_away_and_switch_back` and the park-toast test.
   - The fork flow test, run under a temporary `@private_profile_test`, gets past its approval round and then fails after the simulated restart (`test_console_chat_fork_flow.py:1098`, `assistant_kind 'generic' != None`). Base, with the same marker, failed earlier: its round's marshal raised "App is not running", because later ensures undid the re-point. With the marshal routed to the host as well, base fails at the same restart assertion with the same message, so that failure predates this change. The file is in no PR lane. The marker was removed.
4. TASK-42465: ACs #2 and #4 are ticked with the evidence. #3 is ticked and marked superseded in part (one TASK-41637 test pinned the dropped rule).
5. Wording. The guard comment and ADR-314 now say the harnesses mark that `TldwCli` running and route its `call_from_thread` through their host (`run_on_host`); they no longer say it "runs" on the host. The same correction is in AC #1, these notes and the lesson.
6. Untested plain tests. 120 harness-using tests (164 nodes) in 14 edited harness files outside every PR lane were run under a temporary `@private_profile_test`, on this tree and on a detached base worktree (`fu-r24-harness-base`), with an identical selection.
   - The files: focus mode, gate-1 home/library/Console, prompts controller, setup-lock polish, 30 native-chat-flow, 15 workbench-contract, model chips, probe isolation, 15 command-composer, initial-draft, video capacity, change-turn tracking, hook-review freeze.
   - Both trees: 156 passed, 6 failed and 2 errors, the same nodes and messages. All are pre-existing:
     - three prompts-controller tests: the stale `_CountingResolutionGateway` double has no `cached_context_window`;
     - `test_library_core_loop_modes_are_actionable_without_leaving_library`;
     - `test_conversation_settings_return_releases_transient_modal_mount_failure` (`assert 2 == 1`);
     - `test_console_approvals_chip_activation_focuses_the_decision_not_submit` ("expected the row's decision Select");
     - two `test_chat_screen_context_modal.py` setup errors: its `app_instance` fixture builds the app in the parent, outside the child. Under a temporary `bootstrap_profile` marker instead, both pass on both trees.
   - Every temporary marker was reverted.

   Review-round checks:
   - Soak: the session-close, lifetime, interrupt-rounds, parallel-runs, initial-draft and app-factory files, 20 runs at `-n 4`, two at a time, beside the serial `Tests/Architecture` run. All 20 passed 111/111, in 115-128 s, at load 9-15. The 22 parallel-runs nodes deselected are the local `RecoveryRequired` trap; all 22 also fail on base.
   - `Tests/Architecture`, serial: 1492 passed, 5 skipped. `Tests/CI` at `-n 4`: 1823 passed, 2 skipped. `preflight.sh`: all checks passed.
   - `ruff --select F tldw_chatbook` and on the touched test files: clean. `ruff_gate.py`: rc 0.
   - Lane step "Run admission-sensitive suites", serial: 711 passed, 2 skipped, 3 xfailed.
   - Lane step "Run admission-sensitive suites 4", serial. The first run had 598 passed and 1 failed: `test_console_right_rail.py::test_mounted_inspector_semantic_census_matches_actual_right_rail_order`, with the Inspector row order at index 4 differing. That test then passed 5/5 alone on this tree and 3/3 on base, and its file passed 6/6 serially. A rerun of the step: 599 passed.
   - The `run_on_host` docstring also stops saying the controller "runs on" the host.
<!-- SECTION:NOTES:END -->

