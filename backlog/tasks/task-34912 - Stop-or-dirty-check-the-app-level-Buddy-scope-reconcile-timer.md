---
id: TASK-34912
title: Stop-or-dirty-check-the-app-level-Buddy-scope-reconcile-timer
status: To Do
assignee: []
created_date: '2026-10-13 14:44'
labels:
  - performance
  - ui-perf
  - wave-1-timers
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
BuddyManagement.start_scope_tracking arms an App-level set_interval(0.5s, reconcile_scope) but nothing ever stops it: _scope_timer has no stop call anywhere in the repo. Once armed (any Buddy view presentation or settings save), the app-wide timer sweeps runtime sessions and rewrites sink scope 2x/second for the rest of the session with no change detection, burning idle CPU on every screen. UI/Navigation/buddy_management.py:1010-1014.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Scope timer stops or disarms when no Buddy view or scope consumer remains,Each reconcile_scope tick cheaply no-ops when nothing changed,No app-level 0.5s timer remains armed after the Buddy overlay closes (test)
<!-- AC:END -->
