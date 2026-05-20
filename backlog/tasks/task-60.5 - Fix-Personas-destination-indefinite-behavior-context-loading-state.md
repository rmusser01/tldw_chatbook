---
id: TASK-60.5
title: Fix Personas destination indefinite behavior-context loading state
status: To Do
assignee: []
created_date: '2026-05-20 15:15'
labels:
  - ux
  - hci
  - qa
  - screens
dependencies: []
parent_task_id: TASK-60
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ensure the Personas destination resolves its local behavior-context load into an actionable ready, empty, or error state so users are not left in a permanent loading screen.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Personas screen leaves loading state deterministically in an empty local profile.
- [ ] #2 Personas screen shows an actionable empty or service error state instead of contradictory Ready/loading copy.
- [ ] #3 Attach and open controls reflect actual context availability.
- [ ] #4 Mounted regression waits deterministically without fixed sleeps.
<!-- AC:END -->
