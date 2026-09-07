---
id: TASK-299.1
title: Local reminder CRUD and triggering without server connection
status: Done
assignee:
  - '@macbook-dev'
created_date: '2026-07-19 16:37'
updated_date: '2026-07-19 17:14'
labels:
  - scheduling
  - reminders
dependencies: []
parent_task_id: TASK-299
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the local-only reminder lifecycle so users can create, edit, delete, and trigger reminders while offline. This is AC #2 from TASK-299.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Reminders can be created from the workbench and persisted locally without a server connection
- [ ] #2 Reminders can be edited and deleted from the workbench
- [ ] #3 The scheduler polls for due reminders and dispatches a notification/triggers the reminder handler
- [ ] #4 Local-only operations work when server reachability is 'unreachable'
- [ ] #5 Existing watchlist projections continue to render alongside local reminders
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented in PR #708 (feature/task-299.1-local-reminder-crud → dev). ReminderForm supports one-time/recurring schedules and edit mode; SchedulesWorkbench wires create/edit; TaskDetail wires enable/disable. Targeted pytest: 374 passed; ruff clean.
<!-- SECTION:NOTES:END -->
