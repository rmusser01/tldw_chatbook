---
id: TASK-34930
title: Bound-the-Trajectory-ledger-mount-time-sync-render
status: To Do
assignee: []
created_date: '2026-10-13 14:53'
labels:
  - performance
  - ui-perf
  - wave-3-screen-switch
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Trajectory builds all row specs synchronously at mount whenever the ledger is below WORKER_THRESHOLD = 5000 records (trajectory_screen.py:460-477); above it, a thread worker renders. 5000 is high enough that large-but-under-threshold sessions still pay a full sync render on the navigation path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Mount-time synchronous render bounded by a lower threshold or first-window render plus background fill,No regression for small ledgers (immediate full render below the new bound)
<!-- AC:END -->
