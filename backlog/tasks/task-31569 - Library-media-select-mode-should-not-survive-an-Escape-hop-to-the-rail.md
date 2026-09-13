---
id: TASK-31569
title: Library media - select mode should not survive an Escape hop to the rail
status: To Do
assignee: []
created_date: '2026-09-05 03:23'
labels:
  - library
  - media-ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Select mode survives an Escape hop to the rail, where s no-ops until a second Escape returns focus to Items (self-recovering but confusing; wave 4 PR B Task 2 re-review minor).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Leaving Items focus for the rail exits select mode, or s re-enters it from the rail
- [ ] #2 A test pins the chosen behaviour
<!-- AC:END -->
