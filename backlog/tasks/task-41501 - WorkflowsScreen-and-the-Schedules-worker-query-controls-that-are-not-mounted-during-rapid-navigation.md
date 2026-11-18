---
id: TASK-41501
title: "WorkflowsScreen and the Schedules worker query controls that are not mounted during rapid navigation"
status: To Do
assignee: []
created_date: 2026-11-17 05:27
labels:
- follow-up
- ui
- workflows
- schedules
priority: medium
references:
- tldw_chatbook/UI/Screens/workflows_screen.py
- Tests/UI/test_screen_navigation.py
dependencies:
- TASK-41127
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41127 (libtests, round 15). `WorkflowsScreen._show_status` (introduced by c7fe03ab08) queries controls that are not mounted and loops on `NoMatches`. In the libtests fix round the Schedules worker raised `NoMatches #scheduling-task-table` during tab-switch storms too, so `test_rapid_tab_switch_storm_leaves_no_zombie_widgets` had to stay off its lane.

**Evidence.** `test_overlapping_navigate_requests_complete_in_fifo_order` fails 2 of 3 on the Workflows `NoMatches` loop; `test_rapid_tab_switch_storm_leaves_no_zombie_widgets` fails on the Workflows and Schedules `NoMatches`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A status update that arrives before or after its screen's controls are mounted is dropped or deferred without raising or looping
- [ ] #2 `test_overlapping_navigate_requests_complete_in_fifo_order` passes 3 of 3 under a plain run
- [ ] #3 `test_rapid_tab_switch_storm_leaves_no_zombie_widgets` passes 3 of 3 and is admitted to a PR lane
<!-- AC:END -->
