---
id: TASK-28004
title: >-
  Library media list - Escape from Reader focuses the filter input, swallowing
  Down/Enter
status: Done
assignee:
  - '@claude'
created_date: '2026-09-02 04:10'
updated_date: '2026-09-02 05:38'
labels:
  - library
  - bug
dependencies: []
references:
  - >-
    .impeccable/critique/2026-09-02T04-00-36Z__tldw-chatbook-ui-screens-library-screen-py.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Originally filed as a cursor/marker desync (Enter opened a different row than the visibly marked one). Re-verified 2026-09-02 live on dev tip (worktree media-ux-fixes @ b7e89b6de, tmux scratch-profile run). That desync is GONE - the return-receipt focus machinery keeps marker and selection in sync once the list has focus, and moving the selection auto-loads the item in the Reader. Residual defect: Escape from the Reader (footer "esc focus Items") lands keyboard focus on the "Filter media" INPUT above the rows, so the natural next keystrokes are swallowed - typed characters land in the filter, Down does nothing until Tab moves focus to the rows. Fix direction: land focus on the item ROWS (ideally the currently-loaded row) instead of the filter input.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After Escape from the Reader, Down/Up immediately move the list selection (no keystroke swallowed by the filter input)
- [x] #2 Focus lands on the loaded item's row so the next Down selects (and auto-loads) the adjacent item
- [x] #3 A regression test pins the Escape-then-Down path
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm the Escape branch focuses #library-media-filter\n2. Pinning test: Escape from Reader content focuses the loaded row, next Down moves selection\n3. Helper that prefers the loaded/selected row, falling back to first row then filter\n4. Related reader/media test files green
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: on Escape from the Reader with the Items pane open, the outward-graduation handler focused #library-media-filter (the Filter Input above the rows), so the natural next keystrokes were swallowed - typed characters landed in the filter, Down was inert until Tab. The old wrong-item desync first filed was already gone (return-receipt machinery keeps marker/selection in sync). Fix: new _focus_library_media_items_pane() focuses the loaded/selected item's ROW (matching _selected_media_id, falling back to first row, then the filter only when the list is empty); the Escape handler calls it instead of focusing the filter. Escape-then-Down is now the sequential-review gesture. Tests: test_escape_from_reader_focuses_loaded_row_and_down_advances (Escape lands on the loaded row, Down moves focus to the sibling row - deterministic; the downstream auto-load-on-arrow stays covered by test_arrow_traversal); updated the fake-based escape pins to expect the items-pane landing. Files: UI/Screens/library_screen.py, Tests/UI/test_library_media_reader_flow.py. Whole reader-flow file green (34 passed); isolated stability 6/6.
<!-- SECTION:NOTES:END -->
