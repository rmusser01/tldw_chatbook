---
id: TASK-33242
title: Settings dialogs open and close without a full-screen repaint stall
status: To Do
assignee: []
created_date: '2026-09-28 08:00'
labels:
  - settings
  - theme
  - perf
priority: low
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up from PR #2877's review: opening or closing any Settings dialog (Rename, Delete, Export, leave prompt) repaints the whole screen, ~250 ms at 211x44.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The dominant cost of a Settings dialog open/close is measured and identified
- [ ] #2 The stall is cut to under 100 ms, or a documented reason why not and the best achievable number
<!-- AC:END -->
