---
id: TASK-33076
title: Theme Export lets the user choose the destination
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
Critique #3 P3 (open since critique #2). Export always writes ~/Downloads/<name>_theme.toml. Evidence and file:line causes: Docs/superpowers/qa/2026-09-27-theme-critique-3/findings.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Export asks for a destination defaulting to the Downloads folder
- [x] #2 The chosen path is validated and never written outside it without confirmation
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. export_theme reads the file, then prompts for a destination (RagProfileNameModal) prefilled with ~/Downloads/<name>_theme.toml, validated inline.
2. Validation: absolute path via path_validation, .toml suffix, existing folder (Downloads may be created), not a link/non-regular file, not the themes folder.
3. Keep the overwrite confirmation; tests monkeypatch Path.home and never touch the real Downloads.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
export_theme reads the file, then opens the name/path prompt prefilled with ~/Downloads/<name>_theme.toml. _export_target validates (path_validation.validate_browsing_path via the _typed_path helper shared with Import; .toml suffix; existing folder -- only the default Downloads may be created; not the themes folder; lstat refuses folders, links and non-regular files) inside the dialog and again after dismiss. Existing file -> the existing Overwrite confirmation. Tests monkeypatch Path.home to tmp dirs; the real ~/Downloads is never touched.
<!-- SECTION:NOTES:END -->
