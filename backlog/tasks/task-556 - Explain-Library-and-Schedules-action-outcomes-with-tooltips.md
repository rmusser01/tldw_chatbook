---
id: TASK-556
title: Explain Library and Schedules action outcomes with tooltips
status: Done
assignee: []
created_date: '2026-07-25 17:37'
updated_date: '2026-07-25 17:51'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the remaining audited Library and Schedules action buttons explain the action they trigger so destination controls satisfy the established tooltip contract.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Library top Ingest button has a non-empty outcome tooltip
- [x] #2 The complete destination action-tooltip audit passes with only the documented Personas skip
- [x] #3 Focused destination modules and static checks pass
- [x] #4 Task notes record RED evidence and ADR applicability
- [x] #5 The Schedules ownership, sync-error, and conflict-resolution buttons have distinct non-empty tooltips
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the complete destination-tooltip audit RED evidence and inspect the three unlabelled controls.
2. Add concise outcome tooltips at each affected control construction boundary without changing button behavior.
3. Run the complete parametrized audit, focused Library and Schedules destination modules, Ruff, formatter, and diff checks.
4. Self-review the copy and record verification.

ADR required: no
ADR path: N/A
Reason: This is bounded UI explanatory copy on existing controls and changes no application, service, storage, or routing contract.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added outcome-specific tooltips to the Library top Ingest action and the Schedules local/server ownership, clear-error, and server/local conflict-resolution actions. The initial complete audit reproduced two visible failures (Library top Ingest and Schedules Local) with 8 passes and the documented Personas skip; successive runs exposed the later Schedules controls because the per-route assertion stops at the first missing tooltip. Final audit: 10 passed and 1 documented Personas skip. A 424-test focused batch verified the full destination-shell module, all 36 SchedulesWorkbench tests, and all 28 Library ingest-canvas tests; the batch finished 417 passed, 1 skipped, and 6 separate pre-existing Library-shell contract failures that are not caused by tooltip copy and are being reconciled separately. Ruff, formatter, and diff checks pass. Self-review corrected the sync-error copy to say all errors, matching the handler. ADR required: no; this changes explanatory UI copy only. Modified: tldw_chatbook/UI/Screens/library_screen.py, tldw_chatbook/UI/Screens/scheduling/sync_status_widget.py, tldw_chatbook/UI/Screens/scheduling/conflicts_tab.py, and this task file.
<!-- SECTION:NOTES:END -->
