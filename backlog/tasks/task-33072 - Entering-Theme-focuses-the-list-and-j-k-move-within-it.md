---
id: TASK-33072
title: Entering Theme focuses the list and j/k move within it
status: Done
assignee: []
created_date: '2026-09-27 18:00'
labels:
  - settings
  - theme
priority: low
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 P3. After the rail or Appearance ▸ Open Theme, focus stays outside the list so c/t/e do nothing until F6/Tab; j/k do nothing in the list. Evidence and file:line causes: Docs/superpowers/qa/2026-09-27-theme-critique-3/findings.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Opening Theme from the rail or Appearance focuses the theme list
- [x] #2 j and k move the list highlight
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Bind j/k to cursor down/up on ThemeOptionList (the screen's j/k rail handler already stands aside when focus is in the detail pane).
2. In SettingsScreen._select_category, for Theme with restore_focus, queue a post-swap callback that focuses the theme list -- only if focus is still where the swap put it (the rail row or nothing), so an F6 pressed mid-swap still wins.
3. Tests: rail Enter and Appearance > Open Theme land focus on the list; j/k move the highlight.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
ThemeOptionList binds j/k to cursor down/up (show=False); the screen's j/k rail handler already stands aside outside the rail, so the list wins when focused. SettingsScreen._select_category queues _focus_theme_list behind the pane swap for Theme with restore_focus (rail click/Enter, Open Theme, overview links). It focuses the list only while focus is still on the rail or nowhere, and an F6/Shift+F6 pressed mid-swap clears the intent (_theme_list_focus_pending) -- test_settings_f6_pressed_mid_category_swap_lands_in_the_new_panes caught the first cut, where the queued F6 cycled relative to the list. Tests: test_rail_entry_focuses_the_theme_list_and_c_acts, test_rail_click_and_open_theme_focus_the_theme_list, test_j_and_k_move_the_list_highlight. Files: settings_theme_picker.py, tldw_chatbook/UI/Screens/settings_screen.py, settings.md.
<!-- SECTION:NOTES:END -->
