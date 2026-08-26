---
id: TASK-15666
title: 'busy_fleet_session_count prunes the fleet as a side effect of a read'
status: To Do
assignee: []
created_date: '2026-08-11 21:30'
labels:
  - console
  - agents
  - threading
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`busy_fleet_session_count` calls `fleet_snapshot`, which prunes terminal handles as a side effect, and it is called from the UI thread to build a navigation confirm. A read-shaped method on the UI thread should not mutate coordinator state that worker threads also write. The count itself was fixed in PR 3a-1 Task 6b (it previously reported "0 runs will be killed" and then killed one); this is about how it obtains the number.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The busy count is derived without mutating coordinator state
- [ ] #2 Pruning still happens where it did before (between turns), on the same schedule
- [ ] #3 A test asserts that taking the busy count leaves the handle set unchanged
<!-- AC:END -->
