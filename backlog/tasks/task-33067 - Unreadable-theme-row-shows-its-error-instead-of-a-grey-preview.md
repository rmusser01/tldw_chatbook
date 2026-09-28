---
id: TASK-33067
title: Unreadable theme row shows its error instead of a grey preview
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
Critique #3 P3. Highlighting an unreadable theme file paints every preview row #808080 on #808080 (1:1). Evidence and file:line causes: Docs/superpowers/qa/2026-09-27-theme-critique-3/findings.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An unreadable entry shows its error in place of the preview
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. In ThemePicker._show hide the ThemePreview whenever the highlighted entry has an error (the card-error line above it then stands in its place).
2. Test: highlight an unreadable row -> preview hidden, error shown; a readable row brings the preview back.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
ThemePicker._show hides the ThemePreview whenever the highlighted entry has an error; the existing card-error line directly above it takes its place (no more #808080-on-#808080 block). The preview is only painted while shown. Test: test_unreadable_row_shows_its_error_in_place_of_the_preview (Tests/UI/test_settings_theme_picker.py). Files: tldw_chatbook/Widgets/settings_theme_picker.py, Docs/User_Guide/settings.md.
<!-- SECTION:NOTES:END -->
