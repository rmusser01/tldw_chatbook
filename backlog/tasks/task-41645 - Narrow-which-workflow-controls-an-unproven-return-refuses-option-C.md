---
id: TASK-41645
title: "Narrow which workflow controls an unproven return refuses (option C)"
status: Done
assignee: []
created_date: 2026-11-17 19:05
labels:
- follow-up
- workflows
- console
priority: medium
references:
- tldw_chatbook/Chat/console_workflow_runs.py
- backlog/decisions
dependencies:
- TASK-41320
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Checked on dev 97f8c58c0c; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** TASK-41320 made an unproven control return or metadata read refuse further controls with `workflow_return_unproven`, and the Console says that a restart releases held chats. On 2026-11-17 the owner chose restart-only and asked for the narrowing to be filed. Under option C, only controls that act on the unproven resource refuse; the rest keep working without a restart.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ADR-285 records which controls an unproven return refuses and why
- [x] #2 Controls outside that set work on a run with an unproven return
- [x] #3 Tests cover one refused and one allowed control
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes, a dated ADR-285 amendment recording which controls an unproven return refuses and why.

1. Analysis first: list, per unproven resource (a control return `control_unknown`, which is per run; an unproven metadata read, whose marker is coordinator-wide), which controls refuse today and which the resource itself requires to refuse. If the set turns out to be a product trade-off rather than determined by the resource, stop and report options.
2. Under the read marker, refuse before any effect only the controls whose completion needs a proven workflow metadata read: reconcile (already), and approve, start and revise, whose new driver can only release its chat through a proven terminal publication. Stop, Pause, Resume, quality acceptance and result removal keep working and report "applied" (TASK-41647).
3. Test: under the marker, Approve is refused with `workflow_return_unproven` and changes nothing; Stop on the same workflow still applies.
4. ADR-285 amendment; covering files on base and head.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Analysis (d0486978dd). The two unproven returns hold different resources, so the set that refuses follows from the resource, not from a product choice:
- **Unproven control return** (`control_unknown`) is per run. Pause and Stop on that run refuse ("workflow original control pending"), Recovery refuses with `workflow_return_unproven`, and Approve, Revise and Resume of that workflow refuse because the run is still held. Other workflows and every view are unaffected. Already narrow; no change.
- **Unproven metadata read** sets the coordinator-wide marker, which refuses every `_metadata` read: every view and every command's read-back. So every command committed and then reported the read's error (TASK-41647), and Recovery refused up front. A control acts on this resource when its own completion needs a proven metadata read: Recovery, and Start, Approve (of a phase) and Revise, whose new driver returns its chat only through a proven terminal publication; under the marker the planning or phase would spend work and then hold the chat until restart (shown red below). Stop, Pause, Resume, Accept and Remove need none.

Change: `_refuse_new_driver_if_unproven()` makes Start, Approve (action `approve`) and Revise refuse with `ValueError("workflow_return_unproven")` before any effect while the marker is set. Recovery already refused. The others keep working and report "applied" (TASK-41647). The Setup sheet's Prepare plan, which calls Start without reading a workflow first, is where users meet it: "Prepare plan unavailable · ... Restart Chatbook ...". ADR-285 has a dated amendment with the set and why; the User Guide Recovery paragraph says plan preparation, approval and revision wait for a restart.

Not narrowed (left for the owner): the reads themselves. Letting other workflows' reads or the detail view proceed under the marker means trusting the shared read chain after one read could not prove its storage handles returned; that is a trade-off, recorded in ADR-285 as unchanged. Because of it the detail view still cannot render under the marker, so Stop and Pause, although allowed, are reachable only from code paths, not from its disabled buttons.

Scope note: the owner's "option C" in TASK-41320's decision (2026-11-16) was a different change: in `_native_control`, count an ordinary control refusal whose worker returned, with no committed decision row, returned storage handles and current sources, as a refusal instead of `control_unknown` (reproduce first: Stop racing the end of a phase gets `workflow_authority_changed` and holds the chat until restart). This task's filed text describes the narrowing above, which is what was implemented; that original option C is reported for filing.

Test `test_unproven_read_refuses_approve_but_stop_still_applies` (`approve_under_unproven_read`): a phase awaits review; another chat's read loses a storage handle (the marker); Approve is refused with `workflow_return_unproven` and changes nothing (no run, no primary claim, stored workflow equal, provider calls `["planning"]`); Stop on the same workflow then applies (`workflow_command_applied`, stored `stopped`). Red with the guard taken out (82277899e6's product code): Approve claimed the chat, started its driver and reported `workflow_command_applied`.

Evidence (shared by TASK-41645/41646/41647; branch fu2/wfruns on d0486978dd; shared machine, load 30-96 on 18 CPUs):
- `Tests/Chat/test_workflow_failed_phase_release.py`: base 8/8 (-n 4, load ~35). Head 11/11 inside the covering run below. Two earlier head runs at -n 4 and load 85-90 each failed `test_unproven_final_metadata_read_is_released_only_by_restart` on approve's trailing read becoming unproven (a race in that TASK-41320 scenario; its phase has no preparation, so these changes do not reach it), and one of them also failed `test_phase_failing_with_a_live_worker_keeps_custody_until_reviewed`; each passed when rerun alone.
- 39 covering files (`Tests/Chat/test_workflow_*`, `Tests/Chat/test_console_workflow_*`, `Tests/Agents/test_workflow_*`, `Tests/UI/test_console_workflow_*`, `test_workflow_existing_admission`, `test_workflow_history_controls`, `test_workflow_setup_worker`) at -n 4, base and head running at the same time (load 65-96): base 104 failed / 11 errors / 382 passed; head 107 failed / 11 errors / 382 passed plus the 3 new tests passing. 11 nodes failed only on head. Rerun together on both trees (-n 2 each, load ~67): head 4 failed / 7 passed, base 6 failed / 5 passed, and all 4 head failures also failed on base. No head-only failure remains.
- `Tests/Architecture` serially on head: 3 failed / 1476 passed / 5 skipped. The 3 (`agent_service.py` diagnostic inventory, two `ProjectIndexBindingCard` timer-path checks) fail identically on base d0486978dd; none names a workflow file.
- `./scripts/preflight.sh` passes.

Files: `tldw_chatbook/Chat/console_workflow_runs.py`, `Tests/Chat/test_workflow_failed_phase_release.py`, `backlog/decisions/285-native-phased-agent-workflows.md`, `Docs/User_Guide/console/agent-runs-and-tools.md`.
<!-- SECTION:NOTES:END -->
