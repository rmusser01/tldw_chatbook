---
id: TASK-299.2
title: Bidirectional reminder sync with tldw_server
status: Done
assignee: []
created_date: '2026-07-19 16:37'
updated_date: '2026-07-20 03:39'
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
- [x] #1 Server-connected reminders are pulled into the local DB on sync
- [x] #2 Local create/update/delete mutations are pushed to /api/v1/tasks and queued when offline
- [x] #3 Conflicts are resolved server-wins and surfaced in the workbench
- [x] #4 Sync state (last_pull_at, last_push_at, sync_errors) is updated after each attempt
- [x] #5 Existing local-only reminders remain functional when sync fails
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented TASK-299.2 bidirectional reminder sync across 13 implementation tasks.

Key changes:
- Added typed server-client exceptions and `ServerClientConfig`; hardened `SchedulingServerClient` with retry, typed errors, and idempotency-key stripping.
- Added `SyncCompleted`/`SyncFailed` events for workbench feedback.
- Refactored `SchedulingService` and `SyncEngine` for explicit `owner_id`, network-then-transaction sync, and server-wins conflict resolution.
- Added connection-aware bulk helpers to `ScheduledTasksDB` for efficient sync persistence.
- Built `SyncStatusWidget`, `ConflictsTab`, and wired them into `SchedulesWorkbench` with a sync worker, owner switcher, and conflict-refresh handlers.
- Updated `app.py` to always instantiate `SchedulingServerClient` so the service can be reconnected after login without rebuilding `SchedulingService`.
- Updated ADR-018 with TASK-299.2 decisions on idempotency, retry policy, network-then-transaction boundary, crash-window trade-off, owner identity fallback, runtime-source mapping, and conflict resolution.

Verification:
- `pytest Tests/Scheduling Tests/UI/test_schedules_workbench.py` — 222 passed.
- `ruff check tldw_chatbook/Scheduling tldw_chatbook/UI/Screens/scheduling tldw_chatbook/app.py` — clean.
- `mypy tldw_chatbook/Scheduling tldw_chatbook/UI/Screens/scheduling` — clean.

Implementation plan: `Docs/superpowers/plans/2026-07-19-bidirectional-reminder-sync-implementation-plan.md`.
<!-- SECTION:NOTES:END -->
