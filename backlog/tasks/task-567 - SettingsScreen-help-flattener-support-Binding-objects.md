---
id: TASK-564
title: 'SettingsScreen help flattener: support Binding objects'
status: To Do
assignee: []
created_date: '2026-07-25 07:57'
labels:
  - settings
  - tech-debt
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
SettingsScreen.action_show_workbench_help (541 AC6 fix) flattens only tuple/list BINDINGS entries; a future Binding(...) entry would silently vanish from the F1 help with no test failing. Forward-compat only — all current entries are tuples.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Binding instances are rendered in the screen help output,Regression test covers a mixed tuple+Binding BINDINGS list
<!-- AC:END -->
