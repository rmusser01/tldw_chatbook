---
id: TASK-97
title: Conflict resolution dialog for Notes sync
status: To Do
assignee: []
created_date: '2026-06-11 17:05'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The sync engine's ASK strategy records conflicts and skips the notes; the UI option is now labeled 'Skip and record for review'. Build a real per-conflict prompt: modal showing file version vs app version with keep-file / keep-app / skip choices, reachable from recorded conflicts in the activity log.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 User is prompted per conflict during a sync run with resolution=ask,User can choose file version or app version or skip for each conflict,Chosen resolution is applied and recorded in sync history
<!-- AC:END -->
