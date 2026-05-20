---
id: TASK-60.6
title: Fix Watchlists destination indefinite local snapshot loading state
status: To Do
assignee: []
created_date: '2026-05-20 15:16'
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
Ensure the Watchlists destination resolves its local snapshot load into an actionable ready, empty, or error state so users can understand whether watchlist runs are available.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Watchlists screen leaves loading state deterministically in an empty local profile.
- [ ] #2 Watchlists screen shows an actionable empty or service error state instead of permanent loading copy.
- [ ] #3 Open and Console controls reflect actual run availability.
- [ ] #4 Mounted regression waits deterministically without fixed sleeps.
<!-- AC:END -->
