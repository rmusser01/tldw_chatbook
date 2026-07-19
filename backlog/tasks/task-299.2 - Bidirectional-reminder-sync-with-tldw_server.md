---
id: TASK-299.2
title: Bidirectional reminder sync with tldw_server
status: To Do
assignee: []
created_date: '2026-07-19 16:37'
updated_date: '2026-07-19 16:38'
labels:
  - scheduling
  - reminders
  - sync
dependencies:
  - TASK-299.1
parent_task_id: TASK-299
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wire the Scheduling service and SyncEngine to push local reminder changes to tldw_server and pull server-side reminders down, handling offline buffering and conflicts. This is AC #3 from TASK-299.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Server-connected reminders are pulled into the local DB on sync
- [ ] #2 Local create/update/delete mutations are pushed to /api/v1/tasks and queued when offline
- [ ] #3 Conflicts are resolved server-wins and surfaced in the workbench
- [ ] #4 Sync state (last_pull_at, last_push_at, sync_errors) is updated after each attempt
- [ ] #5 Existing local-only reminders remain functional when sync fails
<!-- AC:END -->
