---
id: TASK-41320
title: "Workflow reconcile still refuses, until restart, a retained run with an open preparation, an unproven control return or an unproven metadata read"
status: In Progress
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
- [ ] #1 A retained run whose only live resource is an open, never-launched preparation can be reconciled: the preparation is closed without effects, the run is recorded as reviewed history, and the chat accepts a new turn, shown by a test that is red on dev
- [ ] #2 For an unproven control return and an unproven metadata read, the chat can be released without a restart through an explicit, recorded user decision, or the task notes and ADR-285 record why only a restart may release it and the UI says so
- [ ] #3 Reconcile still refuses while any launched work, admitted owner or live native thread remains (the existing TASK-40718 tests still pass)
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
