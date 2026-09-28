---
id: TASK-33243
title: Command palette theme switch persists the launch default off the UI thread
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
Follow-up from PR #2877 (TASK-33121 moved the picker's Use write to a worker). The command palette's theme switch still writes the launch default synchronously on the UI thread (~80-150 ms).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A palette theme switch does not block the UI thread for the config write
- [ ] #2 It shares the numbered launch-default write queue, so ordering with picker Use/Revert holds and quit waits for it
<!-- AC:END -->
