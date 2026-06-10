---
id: TASK-83
title: 'Rebuild Notes screen on destination workbench (Notes mode, feature parity)'
status: To Do
assignee: []
created_date: '2026-06-10 03:16'
labels: []
dependencies:
  - TASK-82
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implements the Notes-mode workbench from Docs/superpowers/specs/2026-06-09-notes-workbench-design.md. ADR: none required.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Notes screen renders header + Notes/Sync/Templates mode strip + three-pane workbench
- [ ] #2 Feature parity: search, keyword filter, sort, create blank/from template, import, edit title/content/keywords, auto-save, export/copy, delete, emoji, Use in Console
- [ ] #3 Previously dead buttons (template-create, import, sort-order) work
- [ ] #4 Mode switches preserve unsaved editor content
- [ ] #5 Legacy Notes_Window untouched and functional
- [ ] #6 Notes UI test suites pass
<!-- AC:END -->
