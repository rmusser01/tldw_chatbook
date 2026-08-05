---
id: TASK-1214
title: DestinationHeader.on_mount queries children that may not be mounted
status: To Do
assignee: []
created_date: '2026-07-28 14:45'
labels:
  - bug
  - ui
  - workbench
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
DestinationHeader.on_mount calls sync_state, which query_one()s #workbench-header-title, #workbench-header-subtitle and #workbench-header-status. Those are the header's own compose() children, and on a busy frame they are not mounted yet, so query_one raises NoMatches out of on_mount. Reproduced intermittently (roughly 1 run in 3) against the Console destination header, #console-workbench-header, while running an unrelated Speech test suite that mounts the full app. Pre-existing on dev; not introduced by the Speech redesign branch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Mounting a DestinationHeader never raises NoMatches from on_mount, under repeated runs
- [ ] #2 The header still shows its title, subtitle and status once mounted
- [ ] #3 A regression test mounts the header and fails on the unguarded version
<!-- AC:END -->
