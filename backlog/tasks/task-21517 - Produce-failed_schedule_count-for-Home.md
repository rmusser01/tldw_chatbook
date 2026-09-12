---
id: TASK-21517
title: Produce failed_schedule_count for Home
status: To Do
assignee: []
created_date: '2026-08-31 02:43'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
HomeDashboardInput.failed_schedule_count is a documented dead input (no producer; ladder branch 3 unreachable from real state). Needs a failed-schedules query in Scheduling/db/scheduled_tasks_db.py first — skipped per spec 2026-08-29 decision rule (no cheap query exists today)
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 failed_schedule_count populated from real schedule state,Ladder recover_schedules branch reachable from real state
<!-- AC:END -->
