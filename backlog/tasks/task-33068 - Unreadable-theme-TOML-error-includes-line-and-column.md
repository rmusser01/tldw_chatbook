---
id: TASK-33068
title: Unreadable theme TOML error includes line and column
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
Critique #3 P3. The card says only 'not valid TOML' while the decoder already reports a path-free line/column. Evidence and file:line causes: Docs/superpowers/qa/2026-09-27-theme-critique-3/findings.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The unreadable-file error names the line and column when the parser provides them, and never the path
<!-- AC:END -->
