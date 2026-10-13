---
id: TASK-34931
title: Move-Schedules-results-overlay-and-mark-read-queries-off-the-UI-thread
status: To Do
assignee: []
created_date: '2026-10-13 14:54'
labels:
  - performance
  - ui-perf
  - wave-4-blocking-rebuilds
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clicking the Schedules Results badge runs list_automation_results(limit=200) + count_automation_results + full list_automation_definitions inline on the UI thread before push_screen (schedules_workbench.py:2483); the r mark-read binding reads the unread set inline (:3808) and _push_conflicts_overlay reads get_conflicts inline (:2541). _read_counts in the same file already uses asyncio.to_thread, so this is an inconsistency, not a convention — click/keypress lag scales with backlog size.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Overlay and mark-read query prep runs via to_thread or a worker with handlers only orchestrating,UI stays responsive with a large results backlog (targeted test)
<!-- AC:END -->
