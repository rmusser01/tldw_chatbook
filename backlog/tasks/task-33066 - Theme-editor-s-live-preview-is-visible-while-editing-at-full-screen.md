---
id: TASK-33066
title: Theme editor's live preview is visible while editing at full screen
status: Done
assignee: []
created_date: '2026-09-27 18:00'
labels:
  - settings
  - theme
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 P2. The editor stacks Name, Actions, Palette, Presets, Preview; at 211x44 the preview is fully below the fold and at 235x52 only 3 rows show, so edits cannot be seen without scrolling. Evidence and file:line causes: Docs/superpowers/qa/2026-09-27-theme-critique-3/findings.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 At 211x44 and 235x52 the preview is visible alongside the palette without scrolling
- [x] #2 At 80x24 the editor remains fully reachable
<!-- AC:END -->

## Implementation Plan

1. Wrap Palette+Presets and Live Preview in a two-column Horizontal inside the editor card.
2. Reuse the picker's mechanism: `#settings-workbench.settings-workbench-compact` flips the columns to `layout: vertical` (stacked fallback at <=100 cols).
3. Geometry test at 211x44 and 235x52 (preview fully visible with Primary, no scrolling); keep the 80x24 reachability test green.

## Implementation Notes

- The editor card now wraps Palette+Presets and Live Preview in `#settings-theme-editor-columns` (two 1fr columns, height:auto), so at 211x44 and 235x52 the whole 9-row preview sits beside Primary with the editor at scroll_y 0.
- Deviation from plan: the picker's terminal-width `settings-workbench-compact` class (<=100 cols) flips too late for the editor. At 120x36 the editor is only 52 cols, and side by side the colour inputs collapsed to 3 cols. `SettingsThemeEditor.on_resize` instead sets `-stacked` when the editor is narrower than `TWO_COLUMN_MIN_WIDTH = 100`, and the tcss stacks the columns under `-stacked`.
- Tests: `test_editor_preview_is_visible_beside_the_palette_at_full_screen` (211x44, 235x52; failed 0/9 rows before the change) and `test_editor_stacks_the_preview_when_too_narrow_for_two_columns` (120x36). `test_every_editor_control_is_reachable` (80x24, 190x55) stays green.
- Files: Widgets/settings_theme_editor.py, css/components/_settings_splash_theme.tcss (+ bundle), Tests/UI/test_settings_theme_picker_screen.py, Docs/User_Guide/settings.md.
