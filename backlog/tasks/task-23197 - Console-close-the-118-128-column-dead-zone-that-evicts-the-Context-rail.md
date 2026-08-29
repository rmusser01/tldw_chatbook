---
id: TASK-23197
title: 'Console: close the 118-128 column dead zone that evicts the Context rail'
status: To Do
assignee: []
created_date: '2026-08-29 21:56'
labels:
  - console
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Between 118 and 128 columns the Inspector auto-opens, which trips resolve_console_rail_priority and force-collapses the Context rail to a 13-column stub with no explanation. A one-column resize from 117 to 118 swaps which sidebar the user has. Automatic Inspector opening must not evict a visible Context rail.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Context stays visible across 117 to 135 columns with default preferences
- [ ] #2 An automatic Inspector open never force-collapses a visible Context rail
- [ ] #3 If Context is collapsed by rail priority the reason is visible on the stub
<!-- AC:END -->
