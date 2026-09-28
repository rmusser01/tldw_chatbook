---
id: TASK-32944
title: j/k no longer yank focus out of the Theme editor
status: Done
created_date: 2026-09-24 12:00
assignee:
- '@claude'
labels:
- settings
- keyboard
- theme
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-1373 armed j/k category navigation screen-wide, guarded only against Input/TextArea/Select. Pressing j or k on the Theme editor's tree, a focusable preset swatch or an editor button therefore reached the screen's `on_key` and pulled focus to the category rail, throwing the user out of the editor mid-task. j/k should move between categories only from the rail (or when nothing is focused), never from the content panes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 j/k on the theme Tree or a preset swatch leaves focus in the Theme editor
- [x] #2 j/k from a detail-pane button does not move focus to the rail
- [x] #3 j/k still move between categories from the rail and with nothing focused (landing on the screen)
- [x] #4 j/k still never fire from text inputs, a Select, or an open Select overlay
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Narrow `_jk_category_navigation_blocked()` at the root: allowed only when focus is None or inside `#settings-category-pane`
2. Update the task-1373 pin that pressed j from a detail-pane button (it asserted the behaviour this task removes) and add a Theme-editor regression test
3. Update the user guide key table
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
One guard change in `SettingsScreen._jk_category_navigation_blocked` (every j/k path goes through it): the Input/TextArea/Select and open-overlay checks stay; focus outside the category rail now blocks, and no focus at all still allows. This partly reverses task-1373 AC#1 on purpose. Its goal (no need to focus a category button first) still holds for the rail's filter-adjacent widgets and the landing state; "from anywhere" was what broke the editor.

Tests: the task-1373 pin `test_settings_jk_category_navigation_arms_from_non_category_focus` became `test_settings_jk_category_navigation_is_rail_scoped` (j/k from no focus moves; from `#settings-open-appearance` it does not). New: `test_settings_jk_leaves_theme_editor_focus_alone`. The two existing never-steal tests and the tooltip-mirror test (which presses j from the rail) gained `@private_profile_test` so they run outside CI too.

Files: `tldw_chatbook/UI/Screens/settings_screen.py`, `Tests/UI/test_settings_configuration_hub.py`, `Docs/User_Guide/settings.md`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
