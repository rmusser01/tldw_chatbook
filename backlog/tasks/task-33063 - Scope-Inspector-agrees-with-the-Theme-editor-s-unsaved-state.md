---
id: TASK-33063
title: Scope Inspector agrees with the Theme editor's unsaved state
status: Done
assignee:
  - '@claude'
created_date: '2026-09-27 18:00'
labels:
  - settings
  - theme
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 P2. While editing a theme the inspector's pinned header says 'No unsaved changes' while its body says 'Unsaved theme changes: Yes' and the rail shows 'Theme *'. _category_has_unsaved_changes ignores the theme editor's modified flag. Evidence and file:line causes: Docs/superpowers/qa/2026-09-27-theme-critique-3/findings.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 With unsaved theme edits, every inspector and rail surface reports unsaved changes
- [x] #2 After Save or Discard every surface reports no unsaved changes
- [x] #3 A test covers both states
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. _category_has_unsaved_changes answers theme_editor_modified for THEME, so the inspector header, rail marker and category status all read the one flag (THEME is not a guided Save/Revert category and has no state banner, so no other surface starts offering controls).
2. _refresh_theme_modified_widgets goes through _update_draft_status_widgets(THEME), which rewrites the inspector header and the rail label, then updates the Theme-specific inspector row.
3. Pilot test: edit -> every surface says unsaved; Discard and Save -> every surface says clean.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`_category_has_unsaved_changes(THEME)` returns `theme_editor_modified`, so the inspector header, the rail marker and the category status read one flag (the rail's separate THEME special case is gone). `_refresh_theme_modified_widgets` now goes through `_update_draft_status_widgets(THEME)`, which rewrites the header and rail in place. THEME is not a guided Save/Revert category and has no state banner, so no other surface starts offering controls.

Files: UI/Screens/settings_screen.py, Tests/UI/test_settings_theme_picker_screen.py (parametrised over Discard and Save), Docs/User_Guide/settings.md.
<!-- SECTION:NOTES:END -->
