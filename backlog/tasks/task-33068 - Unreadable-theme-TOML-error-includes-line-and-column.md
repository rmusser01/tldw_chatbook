---
id: TASK-33068
title: Unreadable theme TOML error includes line and column
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
Critique #3 P3. The card says only 'not valid TOML' while the decoder already reports a path-free line/column. Evidence and file:line causes: Docs/superpowers/qa/2026-09-27-theme-critique-3/findings.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The unreadable-file error names the line and column when the parser provides them, and never the path
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Pin the expected card/Import text with a failing test (line and column, no path).
2. Build the reason from TomlDecodeError.lineno/colno in _theme_file_error and in Import's parser branch.
3. Update the tests that pin the bare string.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Unreadable-file card reason and Import's refusal append ' (line N, column M)' built only from TomlDecodeError.lineno/colno (new _toml_position helper in settings_theme_editor.py) -- never str(exc), so no path or file bytes (R16/R39). Pinning tests updated in test_settings_theme_file_api.py, test_settings_theme_import.py, test_settings_theme_picker_screen.py.
<!-- SECTION:NOTES:END -->
