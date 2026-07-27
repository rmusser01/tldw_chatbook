---
id: TASK-1078
title: Remove stale deferred initial-tab test after startup refactor
status: To Do
assignee: []
created_date: '2026-07-27 22:01'
labels:
  - ui
  - tests
  - baseline
dependencies: []
references:
  - Tests/UI/test_screen_navigation.py
  - >-
    backlog/tasks/task-288 -
    Canonicalize-current_tab-through-route-aliases-at-startup.md
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The scoped TASK-944 baseline run found `test_deferred_initial_tab_uses_first_run_home_route` still calling `TldwCli._set_initial_tab`, which was removed by commit 1df0c4cb4. Repair or remove the stale test while preserving current first-run Home routing coverage; do not restore the deleted production method or broaden into TASK-288 alias canonicalization.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The navigation suite no longer calls the removed `_set_initial_tab` method
- [ ] #2 Current first-run Home routing remains covered through the supported startup or resolver path
- [ ] #3 The focused navigation test file passes without production runtime changes
<!-- AC:END -->
