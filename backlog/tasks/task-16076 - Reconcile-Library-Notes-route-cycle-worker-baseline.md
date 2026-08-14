---
id: TASK-16076
title: Reconcile Library Notes route-cycle worker baseline
status: To Do
assignee:
  - '@codex'
created_date: '2026-08-14 04:31'
labels:
  - testing
  - library
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate and repair the independently reproducible Notes fifty-route lifecycle assertion whose final active worker groups differ from the recorded baseline.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The worker-group delta is root-caused,The fifty-route lifecycle test passes without weakening leak detection,Focused static and diff checks pass
<!-- AC:END -->
