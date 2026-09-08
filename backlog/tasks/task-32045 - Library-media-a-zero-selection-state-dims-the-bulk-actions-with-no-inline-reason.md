---
id: TASK-32045
title: >-
  Library media: a zero-selection state dims the bulk actions with no inline
  reason
status: Done
assignee: []
created_date: '2026-09-08 14:37'
updated_date: '2026-09-08 15:42'
labels:
  - library
  - media
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #7 P2. In select mode with nothing selected, Export/Review/Delete are disabled with the '○' marker but no inline reason, while Analyze explains its block. This violates the surface's own 'explain why unavailable' rule that task-31981 established for the analysis actions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 With zero items selected, the disabled bulk actions carry a short inline reason (e.g. 'Select items to enable') the way the analysis block already does
- [x] #2 Selecting an item clears the reason and enables the actions
- [x] #3 A painted pin asserts the reason is present with zero selected and gone once an item is selected
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
In Media select mode with nothing selected, Export/Review/Delete now carry an always-visible inline reason 'Select items to enable.' (id #library-media-select-bulk-reason, reusing task-31981's `.library-media-action-reason` class), mirroring the Analyze block's inline reason. Selecting a row hides the line and enables the actions in the same in-place `_apply_library_row_toggle` patch (no recompose). The line uses `styles.visibility` (NOT `display`) so it reserves its box in both states and the row list below never shifts on a checkbox toggle -- an initial `display` version demonstrably shifted the rows (caught + fixed in-task; verified by `test_every_click_on_a_media_row_toggles_it_in_select_mode`). Excluded when the list failed to load with nothing to select (reuses `_library_media_list_unselectable`, does not touch the failed-load gate). Painted pins at 235x52 and 100x30 via the real compositor (present at 0 selected, nothing painted when hidden), red first. Files: library_media_canvas.py, canvas_sync.py, library_shell_state.py, Tests/UI/test_library_media_render_fixes.py, Docs/User_Guide.
<!-- SECTION:NOTES:END -->
