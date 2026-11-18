---
id: TASK-41647
title: "Workflow commands commit and then report failure while the unproven marker is set"
status: Done
assignee: []
created_date: 2026-11-17 19:05
labels:
- follow-up
- workflows
- console
- ux
priority: low
references:
- tldw_chatbook/Chat/console_workflow_runs.py
dependencies:
- TASK-41320
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Checked on dev 97f8c58c0c; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** This came from TASK-41320's review (M4). While a run carries the unproven marker, a workflow command commits its change and then fails in the trailing `_publish`/get, so the user sees an error for an action that took effect. Stop is the visible case: the run stops, but the command reports failure.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A command that committed reports success, or a distinct "applied, view not refreshed" state, never a plain failure
- [x] #2 Stop keeps working on a run with the unproven marker
- [x] #3 A test covers Stop under the marker
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes, a dated ADR-285 amendment under the TASK-41320 section (what a command that took effect reports when its read-back fails).

1. Red test first (`Tests/Chat/test_workflow_failed_phase_release.py`): with a running phase (held task-1) and the coordinator's unproven-read marker set by another chat's workflow read, Stop takes effect (durable `stopped`, `run.control == "stop"`) but on d0486978dd reports plain `workflow_return_unproven` from its trailing `_publish`.
2. One coordinator helper does every command's trailing publish and read-back. A failure there raises a distinct `workflow_command_applied`, chained from the cause; cancellation still propagates.
3. The Console maps that reason to "<Action> applied · the workflow view could not be refreshed", naming a restart when the cause is the unproven marker, and keeps the action notice when the following refresh fails.
4. ADR-285 amendment and the User Guide Recovery paragraph; mutation check; covering files on base and head.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Every workflow command (start, approve, accept, pause, stop, resume, revise, reconcile, remove results) ended with the same `_publish` + `get` read-back. They now share one helper, `ConsoleWorkflowCoordinator._applied`: if the read-back raises an ordinary exception after the command took effect, it raises `RuntimeError("workflow_command_applied")` chained from that error; cancellation still propagates as itself. The `_metadata_source` calls the eight sites made first are dropped, since `_publish` and `get` make the same check.

Console (`UI/Console_Modules/workflows.py`): `workflow_view_error` maps the reason to "<Action> applied · The workflow view could not be refreshed." plus "Restart Chatbook to read workflows again and release held chats." when the cause is `workflow_return_unproven`, else "Refresh to read the current workflow.". `ConsoleWorkflowStatus._refresh` keeps the action notice under its own message when the refresh after an action fails, so "Stop applied" is not overwritten (one line, no separate UI test).

Stop under the marker (AC #2): Stop's native control and its stored decision never go through the coordinator's metadata reads, so the fix needed is only the report. Under the marker the detail view cannot read a workflow, so its buttons stay disabled; the visible case is a command whose own read-back is the read that becomes unproven. ADR-285 (dated amendment) and the User Guide Recovery paragraph say so.

Test `test_stop_under_an_unproven_read_applies_and_says_so` (`stop_under_unproven_read`): the phase runs with task-1 prepared and its wave call held by a gate; another chat's workflow read loses a storage handle (the real `_remaining_storage_return` fault, scoped to that read's task), which sets the marker; Stop then raises `workflow_command_applied` caused by `workflow_return_unproven`, `run.control == "stop"`, the stored status is `stopped` with a higher revision, the Console copy starts "Stop applied" and names a restart, and once the gate opens the wave refuses (`workflow_authority_changed`) so nothing launches (provider calls `["planning"]`). Shutdown then reports the marker, as designed. A first version held the provider instead of the wave; at load 90 task-1 did not reach the provider within the rig's 10 s, so the gate replaced it.
- Red on c43cfd9c3b (the TASK-41646 commit, without this fix): `AssertionError: workflow_return_unproven` at the applied assertion.
- The durable status of a running execution phase reads `awaiting_approval` (the runtime overlay supplies `running`); `phase._control` accepts Stop from it.

Evidence (shared by TASK-41645/41646/41647; branch fu2/wfruns on d0486978dd; shared machine, load 30-96 on 18 CPUs):
- `Tests/Chat/test_workflow_failed_phase_release.py`: base 8/8 (-n 4, load ~35). Head 11/11 inside the covering run below. Two earlier head runs at -n 4 and load 85-90 each failed `test_unproven_final_metadata_read_is_released_only_by_restart` on approve's trailing read becoming unproven (a race in that TASK-41320 scenario; its phase has no preparation, so these changes do not reach it), and one of them also failed `test_phase_failing_with_a_live_worker_keeps_custody_until_reviewed`; each passed when rerun alone.
- 39 covering files (`Tests/Chat/test_workflow_*`, `Tests/Chat/test_console_workflow_*`, `Tests/Agents/test_workflow_*`, `Tests/UI/test_console_workflow_*`, `test_workflow_existing_admission`, `test_workflow_history_controls`, `test_workflow_setup_worker`) at -n 4, base and head running at the same time (load 65-96): base 104 failed / 11 errors / 382 passed; head 107 failed / 11 errors / 382 passed plus the 3 new tests passing. 11 nodes failed only on head. Rerun together on both trees (-n 2 each, load ~67): head 4 failed / 7 passed, base 6 failed / 5 passed, and all 4 head failures also failed on base. No head-only failure remains.
- `Tests/Architecture` serially on head: 3 failed / 1476 passed / 5 skipped. The 3 (`agent_service.py` diagnostic inventory, two `ProjectIndexBindingCard` timer-path checks) fail identically on base d0486978dd; none names a workflow file.
- `./scripts/preflight.sh` passes.

Files: `tldw_chatbook/Chat/console_workflow_runs.py`, `tldw_chatbook/UI/Console_Modules/workflows.py`, `tldw_chatbook/Widgets/Console/console_workflow_status.py`, `Tests/Chat/test_workflow_failed_phase_release.py`, `backlog/decisions/285-native-phased-agent-workflows.md`, `Docs/User_Guide/console/agent-runs-and-tools.md`.
<!-- SECTION:NOTES:END -->
