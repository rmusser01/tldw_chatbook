---
id: TASK-31798
title: Schedules header permanently shows 'Checking sync status...' when no scheduling server is configured
status: To Do
assignee: []
created_date: '2026-09-05 19:15'
labels:
  - bug
  - ui
  - schedules
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the 2026-09-05 pre-release live UAT sweep (fresh scratch profile, dev tip 8e9d1128d4, real tmux-driven app). With no scheduling server, the Schedules DestinationHeader keeps its compose-time seed 'Checking sync status...' indefinitely (verified 20+ minutes and across navigations) while the same screen's footer correctly says 'Local schedules - no scheduling server connected; sync is off.' Source lead: UI/Screens/scheduling/schedules_workbench.py:586 seeds the label; the refresh at ~line 4315 that would set 'Local only - no server connection' never runs on this path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The header resolves to the local-only status (matching the footer) shortly after mount when no server is configured.
- [ ] #2 Test covering the no-server header path.
<!-- AC:END -->
