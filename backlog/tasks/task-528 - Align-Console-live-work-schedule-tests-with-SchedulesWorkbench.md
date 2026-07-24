---
id: TASK-528
title: Align Console live-work schedule tests with SchedulesWorkbench
status: Done
assignee: []
created_date: '2026-07-24 19:21'
updated_date: '2026-07-24 19:26'
labels: []
dependencies:
  - TASK-527
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep Console live-work schedule handoff coverage on the active SchedulesWorkbench contract after the legacy SchedulesScreen route was retired, without preserving removed labels, inspector copy, or thread-worker assumptions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The live-work module no longer imports or instantiates the retired SchedulesScreen
- [x] #2 Empty, active-run, and digest-output schedule handoffs assert the current SchedulesWorkbench controls
- [x] #3 Obsolete legacy-screen copy, inspector, and off-main-thread assertions are removed
- [x] #4 All schedule-focused cases in the Console live-work handoff module pass
- [x] #5 The merge-base failures and no-ADR decision are documented
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inventory every schedule-specific assertion in the live-work module and compare it with the canonical SchedulesWorkbench tests and scheduling migration history.
2. Replace the retired screen import and input-forwarding call with the active async workbench seam.
3. Keep unique empty/active/digest routing coverage while removing duplicate assertions for retired dynamic labels, inspector copy, and thread workers.
4. Run the schedule-focused live-work slice, Ruff, format, diff checks, and independent review.
5. Document merge-base evidence and the no-ADR decision before completion.

ADR required: no
ADR path: N/A
Reason: This updates stale tests to the already-decided SchedulesWorkbench route and changes no production interface or architecture.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Updated the Console live-work handoff module to instantiate `SchedulesWorkbench`, await its active-work adapter seam, and assert the current generic `Follow in Console` button and recovery tooltip for empty, active-run, and digest-output states. Removed assertions tied only to the retired `SchedulesScreen` dynamic labels, inspector copy, and thread-worker implementation; unique routing and payload coverage remains.

The full UI sweep exposed nine schedule failures whose expectations matched the retired screen. The active workbench suite passes 36/36 and the updated live-work module passes 48/48. A broader destination-tooltip audit still reports the pre-existing missing tooltip on `scheduling-owner-local`; that unchanged control is outside this task's handoff contract.

ADR required: no. This is test alignment to the existing SchedulesWorkbench architecture and changes no service, storage, or runtime boundary.
<!-- SECTION:NOTES:END -->
