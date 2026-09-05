---
id: TASK-31563
title: Repair Prompts adaptive-reader regression coverage
status: Done
assignee: []
created_date: '2026-09-05 01:58'
updated_date: '2026-09-05 02:11'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the remaining Prompts canvas tests after retained Library panes, split owner CSS, and asynchronous browse reconciliation changed test and runtime ownership boundaries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Dirty Prompt navigation coverage expects the established warning while preserving selection.
- [x] #2 Unmount ordering coverage runs inside a valid Textual application context and proves browse authority is revoked before workspace shutdown yields.
- [x] #3 Prompt import settlement keeps the retained Items browse projection synchronized before Undo retries.
- [x] #4 Prompt history contrast coverage loads the Library owner stylesheet.
- [x] #5 Creating and saving a Prompt preserves the mounted block TextArea while refreshing the rail count.
- [x] #6 All 21 focused cases covering the nine reported regressions pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Align stale test expectations and harness context with the current dirty-veto and Textual lifecycle contracts. 2. Load the Library owner stylesheet in direct Prompts canvas geometry harnesses. 3. Trace and fix retained Items/Work synchronization races that leave imports stale or remount the editor after create. 4. Run the 21 focused cases, adjacent targeted coverage, Ruff, and diff checks. ADR required: no. ADR path: N/A. Reason: The work repairs regressions within the adaptive-reader ownership contract already established by ADR-086; it does not introduce a new boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Aligned the Prompts test harness with the current Library owner CSS, dirty-navigation notification, and Textual application lifecycle contracts. Kept the retained Prompts Items projection synchronized without remounting its independently owned Work editor, and prevented unrelated broad Library snapshot failures from replacing the Prompts canvas, which has its own scope service and error handling. The 21 focused regression cases pass, as does the adjacent dedicated Conversation error-canvas test; Ruff passes for the modified test module, the Library screen compiles, and `git diff --check` is clean. Modified `Tests/UI/test_library_prompts_canvas.py` and `tldw_chatbook/UI/Screens/library_screen.py`. ADR required: no; ADR-086 already governs the repaired adaptive-reader ownership boundary.
<!-- SECTION:NOTES:END -->
