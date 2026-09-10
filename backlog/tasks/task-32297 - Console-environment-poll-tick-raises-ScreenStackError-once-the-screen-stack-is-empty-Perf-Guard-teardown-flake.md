---
id: TASK-32297
title: >-
  Console environment poll tick raises ScreenStackError once the screen stack is
  empty (Perf Guard teardown flake)
status: To Do
assignee: []
created_date: '2026-09-10 20:24'
labels:
  - library
  - test-health
  - console
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Chat/Console screen's environment poll timer can fire after the app's last screen has been popped during teardown. `_poll_console_environment` reads `self.app.screen`, which raises `ScreenStackError('No screens on stack')` on an empty stack, so the Perf Guard tour tests (`test_destination_tour_stays_under_switch_budgets`, `test_fastpath_computes_identical_styles_for_every_node`) fail intermittently — three different PRs hit it on 2026-09-10 (#2569, #2571, #2577), each needing a manual rerun. The owner's `rail_open_accessor` lambda carries the same read.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The poll tick and the rail-open accessor treat an empty screen stack as 'not the active screen' and never raise
- [x] #2 A unit test pins the empty-stack tick as a no-op and is red without the guard
<!-- AC:END -->
