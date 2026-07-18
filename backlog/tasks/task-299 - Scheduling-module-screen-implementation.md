---
id: TASK-299
title: Scheduling module + screen implementation
status: To Do
assignee:
  - '@macbook-dev'
created_date: '2026-07-18 23:48'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Build a hybrid local/server scheduling module and TUI workbench in tldw_chatbook, porting tldw_server's unified scheduled-tasks concepts while preserving existing Console-follow and screenshot QA behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Existing schedules destination regressions and screenshot QA still pass after replacement,Reminders can be created/edited/deleted/triggered locally without a server connection,Server-connected reminders sync bidirectionally with tldw_server /api/v1/tasks,Automation definitions can be created/previewed/paused/resumed/archived locally,Watchlist jobs are visible in the workbench as read-only projections
<!-- AC:END -->
