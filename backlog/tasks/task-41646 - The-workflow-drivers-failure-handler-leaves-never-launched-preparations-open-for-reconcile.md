---
id: TASK-41646
title: "The workflow driver's failure handler leaves never-launched preparations open for reconcile"
status: Done
assignee: []
created_date: 2026-11-17 19:05
labels:
- follow-up
- workflows
priority: low
references:
- tldw_chatbook/Chat/console_workflow_runs.py
dependencies:
- TASK-41320
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Checked on dev 97f8c58c0c; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** TASK-41320 made reconcile close a retained run's never-launched, unwaved preparation, so the chat accepts a new turn. The driver's own failure handler sees that same preparation first but leaves it open, so the run stays retained until a reconcile happens to run.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The driver handler closes a never-launched preparation itself, recording the same review row reconcile records
- [x] #2 A test fails a run before launch and asserts that the chat accepts a new turn without reconcile
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (the ADR-285 TASK-41320 section already allows closing a never-launched preparation; this moves the same close earlier, into the driver).

1. Red test first in `Tests/Chat/test_workflow_failed_phase_release.py`: the wave call refuses (ordinary exception) after task-1 is prepared and before any wave binds it; assert the run is not retained, the preparation is closed without launch or dispatch, the workflow is the resumable failed-phase state `blocked_phase` reaches, and the chat accepts a new turn with no reconcile. Red on d0486978dd: the open preparation counts as live, so the run is retained.
2. In `_execute`'s TASK-40718 handler, close the run's never-launched, unwaved preparations (the same runtime-state condition and the same `close()` reconcile uses, shared as one helper) before the live check. The existing retirement tail then settles the phase as failed and releases the chat.
3. The TASK-41320 scenarios that need a retained run with an open never-launched preparation now get it from a cancelled wave call (a BaseException the handler deliberately does not treat as an ordinary failure).
4. Mutation check (drop the new close loop: the new test is red), covering workflow files on base and head.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The driver's TASK-40718 failure handler in `_execute` now closes the run's never-launched preparations before its live check: each item in `run.preparations` that runtime state shows was never enrolled in a wave and never launched (`_wave` unset, empty `_native_launch`). It is the same condition and the same `close()` reconcile uses (TASK-41320), now one helper, `ConsoleWorkflowCoordinator._unlaunched`, that reconcile calls too. `close()` rechecks the launch state under the task lock. With nothing else live, the existing retirement tail settles the phase as `failed`, publishes and returns the chat's claim.

AC #1, "recording the same review row reconcile records", read as: the same check and close, ending in the record TASK-40718 already gives a failed phase whose work all returned (the one `blocked_phase` gets: phase `failed`, workflow resumable, shown as `paused`). Reconcile's own rows (`record_returned_execution` marking the phase uncertain, then `resolve_recovery` closing the workflow) carry the user's explicit "reviewed prior effects" decision, which the driver does not have; recording them here would close a never-launched workflow as reviewed history instead of leaving it resumable. Flagged for the coordinator in the report.

Tests (`Tests/Chat/test_workflow_failed_phase_release.py`):
- New `test_phase_failing_before_launch_returns_its_chat_without_reconcile` (`failed_before_launch`): the wave call raises an ordinary error after task-1 is prepared. It asserts the preparation is closed by the driver, nothing launched or dispatched (provider calls `["planning"]`, no executions), no reconcile command ran, the workflow reads `paused`, and the chat accepts a new turn. Red on d0486978dd (`assert item._closed` failed; the teardown then raised `workflow_original_shutdown_unresolved`, read from the child's full log).
- The three TASK-41320 scenarios that need a retained run with an open never-launched preparation (`unlaunched_preparation`, `unproven_control`, `unproven_earlier_read`) now get it from a cancelled wave call: the handler takes ordinary exceptions only, so a `CancelledError` still retains the run. With only that test change they pass on d0486978dd (3/3), so the helper change is neutral.
- Mutation: the close loop is the only behavioural change; without it (d0486978dd plus the new test) the new test is red, as above.

Evidence (shared by TASK-41645/41646/41647; branch fu2/wfruns on d0486978dd; shared machine, load 30-96 on 18 CPUs):
- `Tests/Chat/test_workflow_failed_phase_release.py`: base 8/8 (-n 4, load ~35). Head 11/11 inside the covering run below. Two earlier head runs at -n 4 and load 85-90 each failed `test_unproven_final_metadata_read_is_released_only_by_restart` on approve's trailing read becoming unproven (a race in that TASK-41320 scenario; its phase has no preparation, so these changes do not reach it), and one of them also failed `test_phase_failing_with_a_live_worker_keeps_custody_until_reviewed`; each passed when rerun alone.
- 39 covering files (`Tests/Chat/test_workflow_*`, `Tests/Chat/test_console_workflow_*`, `Tests/Agents/test_workflow_*`, `Tests/UI/test_console_workflow_*`, `test_workflow_existing_admission`, `test_workflow_history_controls`, `test_workflow_setup_worker`) at -n 4, base and head running at the same time (load 65-96): base 104 failed / 11 errors / 382 passed; head 107 failed / 11 errors / 382 passed plus the 3 new tests passing. 11 nodes failed only on head. Rerun together on both trees (-n 2 each, load ~67): head 4 failed / 7 passed, base 6 failed / 5 passed, and all 4 head failures also failed on base. No head-only failure remains.
- `Tests/Architecture` serially on head: 3 failed / 1476 passed / 5 skipped. The 3 (`agent_service.py` diagnostic inventory, two `ProjectIndexBindingCard` timer-path checks) fail identically on base d0486978dd; none names a workflow file.
- `./scripts/preflight.sh` passes.

Files: `tldw_chatbook/Chat/console_workflow_runs.py`, `Tests/Chat/test_workflow_failed_phase_release.py`.
<!-- SECTION:NOTES:END -->
