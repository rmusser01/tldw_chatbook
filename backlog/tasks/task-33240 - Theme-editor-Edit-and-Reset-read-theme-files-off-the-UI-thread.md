---
id: TASK-33240
title: Theme editor Edit and Reset read theme files off the UI thread
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
Follow-up from the TASK-33078 tail wave (PR #2877). Opening a saved theme in the editor (Edit) and Reset still resolve the theme file through the backup-scoped themes-folder scan on the UI thread (~210 ms with 50 saved themes), unlike the file actions that TASK-33078 moved to a worker. File operations must stay inside SettingsThemeEditor's raw backup scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With 50 saved themes, Edit and Reset never block the UI thread for more than 100 ms
- [ ] #2 The editor shows the correct theme data once the read lands, and a stale read never overwrites a newer editor session
<!-- AC:END -->
