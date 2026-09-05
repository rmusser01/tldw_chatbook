---
id: TASK-31582
title: Library media bulk Analyze - the run should survive leaving Library
status: To Do
assignee: []
created_date: '2026-09-05 03:24'
labels:
  - library
  - media-ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The bulk Analyze worker (task-28007) is screen-owned and is cancelled when the user leaves Library; wave 4 PR D ships an honest unmount notify with a resume hint instead. An app-owned run whose receipt re-renders on return needs the receipt state moved off the screen instance.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Leaving Library does not cancel a running analysis
- [ ] #2 Returning to Media shows the receipt with current or final counts
- [ ] #3 The unmount notify is removed
<!-- AC:END -->
