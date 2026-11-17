---
id: TASK-41320
title: "Workflow reconcile still refuses, until restart, a retained run with an open preparation, an unproven control return or an unproven metadata read"
status: Done
created_date: 2026-11-16 19:55
labels:
- follow-up
- agents
- workflows
- console
priority: medium
references:
- tldw_chatbook/Chat/console_workflow_runs.py
- tldw_chatbook/Agents/workflow_run_service.py
- backlog/decisions/285-native-phased-agent-workflows.md
- https://github.com/rmusser01/tldw_chatbook/pull/3543
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the 2026-11-13..16 burn-down. These are the limits TASK-40718 (PR #3543) recorded. Checked on dev 5e3c8b2bd4 (2026-11-16) by reading the code; line numbers are that tree's, and paths are under `tldw_chatbook/`.

**Why.** TASK-40718 lets a user close a failed native workflow phase as reviewed history once its work has returned, so the chat is usable again without a restart. Three retained states still refuse that reconcile with `workflow_resources_still_live`. In each, the chat keeps its intake claim, new turns are refused ("A turn is already preparing."), and only a restart frees it.

**Evidence.**
- **An open, never-launched preparation** counts as live work. `WorkflowRunService._has_live_workflow_resources` (`Agents/workflow_run_service.py:2267-2300`) returns True for any entry in `runtime._workflow_preparations` for the workflow (`:2276-2284`), whether or not it ever launched. Reconcile does not close such preparations.
- **An unproven control return.** `ConsoleWorkflowCoordinator.reconcile` (`Chat/console_workflow_runs.py:2426-2492`) refuses at `:2446-2460` when `run.control_unknown` is set, or `run.control_return` exists but is not set.
- **An unproven metadata read.** It refuses at the same place when the run's final-metadata cell's `disposed` future is not done, was cancelled, raised, or did not return `True`. App shutdown also refuses while that cell's marker stays in `_metadata_cells` (`:2658-2669`, `workflow_original_metadata_shutdown_unresolved`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A retained run whose only live resource is an open, never-launched preparation can be reconciled: the preparation is closed without effects, the run is recorded as reviewed history, and the chat accepts a new turn, shown by a test that is red on dev
- [x] #2 For an unproven control return and an unproven metadata read, the chat can be released without a restart through an explicit, recorded user decision, or the task notes and ADR-285 record why only a restart may release it and the UI says so
- [x] #3 Reconcile still refuses while any launched work, admitted owner or live native thread remains (the existing TASK-40718 tests still pass)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes, for AC #2 only. Either outcome amends ADR-285 (and ADR-356 rule 6 if the chat is released without a restart). AC #1 needs no ADR: it reuses the existing preparation close.

Evidence (base fbf1c9ea74 = dev 5e3c8b2bd4 + filing commits; same line numbers as the description):
- AC #1 reproduced. `unlaunched_preparation` in `Tests/Chat/test_workflow_failed_phase_release.py`: the wave call refuses before binding task-1. Then task-1's preparation is open, never enrolled in a wave (`_wave is None`) and never launched (`authorization._native_launch == []`). The driver's TASK-40718 live check counts that preparation, so the run is kept `recovery_required` and a new turn is refused. Reconcile then raises `workflow_resources_still_live` from `record_returned_execution`, and shutdown raises `workflow_original_shutdown_unresolved`. The other four tests in the file pass on the same tree.
- (b) and (c): in both retained states, the worker that did the control or the read has already returned. `run_preparation_read` waits for its producer future even when the caller is cancelled. `control_return` is set and `disposed` is resolved only after that wait. An unset `control_return` or an unresolved `disposed` means the work is still in flight, and reconcile must keep refusing it. Three things stay unknown: the outcome (the control's decision transaction committed or rolled back; the metadata read is read-only and always rolls back), whether the thread-local storage handles returned (`_remaining_storage_return`, plus `worker_guard`'s close), and whether the sources stayed current. The common control case is an ordinary refusal from `phase._control`, for example `workflow_authority_changed` when Stop races the end of a phase; it is recorded as unknown because `returned` is empty. The marker for an unproven metadata read also blocks every later metadata read in the coordinator (`_metadata`, :902-906), so reconcile's own trailing `_publish`/`get` would fail too. The UI shows `workflow_resources_still_live` as "Work or cleanup is still active. Wait, then refresh." (`UI/Console_Modules/workflows.py:151-152`). For these states, waiting never helps.

1. AC #1 (after the owner confirms): in `ConsoleWorkflowCoordinator.reconcile`, on the execution branch after the control and metadata checks and before `record_returned_execution`, close each `run.preparations` item that runtime state shows was never enrolled or launched (`item._wave is None and not item.authorization._native_launch`). `close()` rechecks launch state under the task lock. A wave-enrolled or launched item is left alone, so the service's live check still refuses with `workflow_resources_still_live` (AC #3). A retry is idempotent because `close()` returns early once the item is closed. Mutation check: drop the loop, and the new test fails at reconcile.
2. AC #2: carry out the option the owner picks (see the report). The default is option A: reconcile refuses these two states with a distinct reason, and the UI says "restart Chatbook to release this chat" instead of "wait, then refresh". ADR-285 and these notes record why.
3. AC #3: run the existing TASK-40718 tests unchanged, plus the covering files on base and head.
4. Preflight, then `Tests/Architecture` serially.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Owner decisions (2026-11-16): AC #1 as planned. AC #2 is option A: only a restart releases the chat, reconcile refuses with a distinct reason, and the UI and ADR-285 say why. Follow-up C (count a proven control refusal as a refusal) and closing never-launched preparations in the driver's TASK-40718 handler are filed separately.

- **AC #1.** On the execution branch, `ConsoleWorkflowCoordinator.reconcile` runs its control and metadata checks first. It then closes each `run.preparations` item that runtime state shows was never enrolled in a wave or launched (`_wave is None` and an empty `authorization._native_launch`), and records the review as before. `close()` checks launch state again under the task lock; it is the same close the driver's own retirement tail uses. A wave-bound or launching item stays open, and `record_returned_execution`'s live check refuses with `workflow_resources_still_live`. Trade-off: if that record step refuses because some other resource is still live, the preparations closed just before stay closed. That close only released resources, and a retry is idempotent.
- **AC #2.** The old single refusal is split in two. Work still in flight (an unset `control_return`, or a `disposed` future that is not done) still refuses with `workflow_resources_still_live`, because waiting can resolve it. A control whose worker returned unproven (`control_unknown`), or a final read whose `disposed` result is not a proven `True`, refuses with `workflow_return_unproven`. The Console then shows "Recovery unavailable · Chatbook could not confirm that a Pause, Stop or background check finished cleanly. Restart Chatbook to release this chat." (wording after the review round; see below). The new ADR-285 section "Reconciling retained execution runs (TASK-41320)" records the reasoning. In short: the worker has returned (`run_preparation_read` waits for it), but its storage handles, its sources and the control's in-memory signals are unproven, and an unproven read keeps refusing every later workflow metadata read anyway. The User Guide's Recovery paragraph says the same.
- **AC #3.** The four TASK-40718 tests are unchanged and pass. New checks in `unlaunched_preparation` first make the item wave-bound and then mark a launch in progress. Each time, reconcile refuses as live, records nothing and leaves the item open.

Tests (`Tests/Chat/test_workflow_failed_phase_release.py`, native rig in a private child):
- `unlaunched_preparation` (red on base fbf1c9ea74): the wave call refuses before binding task-1. Reconcile now closes the preparation; nothing is launched or dispatched (provider calls stay `["planning"]`). The run is stored `closed` with the review revision, and the chat accepts a new turn. On base, reconcile raised `workflow_resources_still_live` from `record_returned_execution`, and shutdown raised `workflow_original_shutdown_unresolved`.
- `unproven_control`: a Stop on the retained run whose `phase._control` raises an ordinary refusal leaves `control_unknown` set with `control_return` set. Reconcile refuses with `workflow_return_unproven` and records nothing, the Console names a restart, and the chat stays held. Shutdown reports `workflow_original_shutdown_unresolved` (restart-only by design).
- `unproven_final_read`: in a blocked phase, the terminal publication's read loses a storage handle after settlement, so `disposed` resolves `False`. The assertions are the same, and shutdown reports `workflow_original_metadata_shutdown_unresolved`.
- Mutation checks, one at a time on the committed head, each restored afterwards:
  - Without the close loop, the final reconcile raises `workflow_resources_still_live`.
  - Without the `_wave` clause, it raises `PermissionError: workflow close wave changed`.
  - Without the launch clause, it raises `PermissionError: workflow launch return has not completed`.
  - With the old reason, both unproven tests fail on the reason.
  - Without the UI branch, the message has no restart.

Evidence (shared machine, load 50-103 on 18 CPUs):
- The new file passes 7/7 on head.
- 10 covering files (`test_workflow_failed_phase_release`, `test_workflow_run_service`, `test_workflow_recovery`, `test_workflow_recovery_cancellation`, `test_workflow_coordinator_ownership`, `test_workflow_planning_native`, and UI `test_workflow_history_controls`, `test_console_workflow_navigation`, `test_workflow_existing_admission`, `test_console_workflow_command`): base 30 failed / 90 passed, head 37 failed / 86 passed. Head has the 3 new tests, and they pass.
- 7 nodes failed only on head, all planning-native, recovery-cancellation or history UI nodes, none of which reaches the changed branch. Rerun: base passed 7/7; head passed 3 and timed out on 4 at load 95-103. Those 4 then ran serially on both trees at the same time and passed 4/4 on each.
- Preflight passes.
- Tests/Architecture, serial on head (load 55-88): 6 failed, 1415 passed. Three are the known base failures: the `agent_service.py` diagnostic inventory and the two `ProjectIndexBindingCard` timer-path checks. Those three fail identically on base fbf1c9ea74. The other three (`test_task_15103_review_ledger_*`) hit the 300 s pytest timeout. Run alone, they pass on head (3/3) and on base (3/3).
- The shared wrapper `scratchpad/pr3480/run-pytest.sh` disappeared at about 00:10 on 2026-11-17, after the Architecture run started. The comparison reruns used a byte-identical copy at `scratchpad/fu41320/run-pytest.sh`.

Files: `tldw_chatbook/Chat/console_workflow_runs.py`, `tldw_chatbook/UI/Console_Modules/workflows.py`, `Tests/Chat/test_workflow_failed_phase_release.py`, `backlog/decisions/285-native-phased-agent-workflows.md`, `Docs/User_Guide/console/agent-runs-and-tools.md`.
Review round (independent Opus review of acfb7f7ed1, "ready after fixes"). All findings were fixed; each new test was red on acfb7f7ed1 first.
- **Important 1: the restart copy never showed for an unproven read.** While a False marker was registered, `_metadata` refused every read with "workflow original metadata return unproven". The detail view reads through `coordinator.get`, so it showed "Review unavailable · RuntimeError. Refresh to read the current workflow." (red). Now:
  - That refusal, and the preceding-read refusal, raise `workflow_return_unproven`.
  - `_refresh` drops its "Refresh to read" suffix for that reason.
  - `unproven_final_read` opens the real `ConsoleWorkflowStatus` over the real coordinator and asserts the restart copy, with no "Refresh". It uses a bare `App` host: under the full Console harness, `ConsoleWorkspaceTree` failed to compose while the chat held a run (`KeyError: No 'conversation-attention-running' key in COMPONENT_CLASSES`). That bug is reported for filing.
- **Important 2: an earlier unproven read released the chat.** Reconcile checked only `run.final_metadata`. New scenario `unproven_earlier_read`: a retained run's ordinary `get` loses a storage handle, leaving the marker `[False]` with `final_metadata` None. On acfb7f7ed1, reconcile closed the preparation and recorded the review, then raised from its trailing `_publish` (red). Reconcile now refuses with `workflow_return_unproven` before changing anything when any registered read resolved unproven, using the same condition `_metadata` refuses on, shared as `_metadata_unproven()`. The preparation stays open, nothing is stored, and the chat stays held.
- **Minor 1: a stale reconcile closed preparations.** It closed never-launched preparations before `record_returned_execution` checked the revision. When there is anything to close, reconcile now compares the displayed revision with `service.get` first. `unlaunched_preparation` asserts that a stale reconcile raises `revision_conflict` and leaves the item open (red on acfb7f7ed1). The check is skipped when nothing would close, so TASK-40718's racing-launch test still reaches the record step's own revision check; an unconditional pre-check failed it (`launches == []`).
- **Minor 2: copy.** Plain wording, as above. It contains no markup brackets.
- **Minor 3: synthetic checks.** The docstring now says the wave-bound and launching checks set that state by hand, and that real launched, wave-bound coverage is TASK-40718's `live_worker_failure`.
- **Test staging.** The final read's fault now trips only on the driver's own read after settlement, identified by its registered read task. One loaded run had tripped approve's trailing `get` instead.
- **Mutation checks**, each restored afterwards:
  - Old refusal text in `_metadata`: the view shows "Refresh to read".
  - Suffix always appended: "…this chat.. Refresh to read".
  - No global reconcile check: reconcile changes state, then raises `RuntimeError` from `_publish`.
  - No revision pre-check: the stale reconcile closes the item.
- **Files added this round:** `tldw_chatbook/Widgets/Console/console_workflow_status.py`. ADR-285 and the User Guide wording now cover any unproven metadata read.

Review-round evidence (load 50-132):
- The new file passes 8/8 on head.
- 11 covering files (the 10 above plus UI `test_workflow_setup_worker`, because the view changed): base 30 failed, head 21 failed / 114 passed. `test_workflow_setup_worker` passes 11/11 on base.
- Four history UI nodes failed only on head, all subprocess timeouts at load up to 132. Run serially on both trees at the same time, they pass 4/4 on each.
- The new scenarios do not exist on fbf1c9ea74. Their red runs on acfb7f7ed1 and the mutation checks above are the base evidence.
- Preflight passes.
- No Tests/Architecture file names the changed modules. The previous round's full serial run showed only the 3 known base failures. `test_timer_path_static_update_inventory.py` on head fails only the 2 known `ProjectIndexBindingCard` nodes, the same as base; no workflow view appears in its findings.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
