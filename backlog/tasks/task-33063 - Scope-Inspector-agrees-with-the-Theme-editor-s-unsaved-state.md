---
id: TASK-33063
title: Scope Inspector agrees with the Theme editor's unsaved state
status: To Do
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
Critique #3 P2. While editing a theme the inspector's pinned header says 'No unsaved changes' while its body says 'Unsaved theme changes: Yes' and the rail shows 'Theme *'. _category_has_unsaved_changes ignores the theme editor's modified flag. Evidence and file:line causes: Docs/superpowers/qa/2026-09-27-theme-critique-3/findings.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With unsaved theme edits, every inspector and rail surface reports unsaved changes
- [ ] #2 After Save or Discard every surface reports no unsaved changes
- [ ] #3 A test covers both states
<!-- AC:END -->
