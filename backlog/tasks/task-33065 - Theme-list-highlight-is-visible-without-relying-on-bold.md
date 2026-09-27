---
id: TASK-33065
title: Theme list highlight is visible without relying on bold
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
Critique #3 P2. The focused list's highlighted row differs from the list background by 1.10:1 (light) and 1.12:1 (dark); only bold text marks it among bold headers and markers. Evidence and file:line causes: Docs/superpowers/qa/2026-09-27-theme-critique-3/findings.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The highlighted row's fill contrasts at least 3:1 with the list background under Textual Dark and Textual Light
- [ ] #2 The contrast test measures the fill, not just bold
<!-- AC:END -->
