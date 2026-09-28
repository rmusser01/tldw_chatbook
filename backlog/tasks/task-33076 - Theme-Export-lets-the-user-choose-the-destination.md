---
id: TASK-33076
title: Theme Export lets the user choose the destination
status: To Do
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
- [ ] #1 Export asks for a destination defaulting to the Downloads folder
- [ ] #2 The chosen path is validated and never written outside it without confirmation
<!-- AC:END -->
