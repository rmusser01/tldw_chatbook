---
id: TASK-33121
title: Use writes the launch default without blocking the UI thread
status: To Do
assignee: []
created_date: '2026-09-28 00:00'
labels:
  - settings
  - theme
priority: low
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up from TASK-33075 profiling: the picker's Use persists the launch default synchronously, ~140 ms on the UI thread per Use.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Pressing Use does not block the UI thread for the config write
- [ ] #2 A failed write is still reported to the user
- [ ] #3 The launch default is persisted before the app can exit
<!-- AC:END -->
