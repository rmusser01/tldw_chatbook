---
id: TASK-31825
title: Wire DestinationHeader's dormant compact density to a height-based trigger
status: To Do
assignee: []
created_date: '2026-09-06 14:15'
labels:
  - ui
  - responsive
  - workbench
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up from task-31419's real-shell measurement (2026-09-06, 80x24, scratch profile). The measurement corrected PR-4 Task 6's attribution: of the claimed 13 chrome rows, only 4 are true app shell (MainNavigationBar 3, AppFooterStatus 1 -- both deliberate/minimal); the destination header (5 rows), scheduler liveness (1) and status strip (4) are composed inside SchedulesWorkbench itself. The one real narrow-terminal lever found: the shared DestinationHeader widget (UI/Workbench/workbench_widgets.py:~164) already ships a density="compact" CSS rule that NO caller ever triggers, and no height-based responsive logic exists anywhere in the workbench layer. Wiring a height-based trigger there would reclaim rows for every (~12) workbench screen that uses the widget -- a shared-widget-layer change, not UI/Navigation/ and not a per-screen override. Full measurement in task-31419's Implementation Notes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 DestinationHeader renders its compact density automatically below a height threshold, via the shared workbench-widget layer (no per-screen overrides)
- [ ] #2 Every workbench screen using DestinationHeader benefits without per-screen changes, verified on at least Schedules plus one other
- [ ] #3 Geometry is asserted with the bundled stylesheet (compact rows measurably fewer at 80x24; normal density unchanged at standard sizes)
- [ ] #4 Tests/UI/test_schedules_responsive_floor.py stays green (minus the known pre-existing dev red)
<!-- AC:END -->
