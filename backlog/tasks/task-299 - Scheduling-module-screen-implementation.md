---
id: TASK-299
title: Scheduling module + screen implementation
status: In Progress
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
- [x] #1 Existing schedules destination regressions and screenshot QA still pass after replacement
- [ ] Reminders can be created/edited/deleted/triggered locally without a server connection
- [ ] Server-connected reminders sync bidirectionally with tldw_server /api/v1/tasks
- [ ] Automation definitions can be created/previewed/paused/resumed/archived locally
- [ ] Watchlist jobs are visible in the workbench as read-only projections
- [ ] Watchlist checks execute under the unified scheduler (Phase 5)
<!-- AC:END -->

## Implementation Plan

1. Phase 0: API contract spike (server reminder endpoints, automation-definition endpoints).
2. Phase 1: Data layer — config helpers, Pydantic models, schema DDL, `ScheduledTasksDB` CRUD, migrations.
3. Phase 2: Control plane — `SchedulingServerClient`, `SyncEngine`, `SchedulingService`, watchlist projection.
4. Phase 3: Scheduler — `SchedulerLoop`, `ReminderHandler`, wire into app lifecycle.
5. Phase 4: Screen workbench — TCSS, forms, `SchedulesWorkbench`, task list/detail/inspector, screen registry.
6. Phase 5: Watchlist scheduler migration — implement `WatchlistCheckHandler`, feature flag, dual-run validation, remove old `SubscriptionScheduler`.
7. Phase 6: Full test suite, lint/type checks, backlog close-out.

## Implementation Notes

- ADR required: yes
- ADR path: `backlog/decisions/018-local-server-hybrid-scheduled-tasks.md`, `backlog/decisions/019-watchlist-scheduler-migration.md`
- Reason: ADR-018 establishes the local/server hybrid storage and sync policy; ADR-019 records the Phase 5 decision to migrate watchlist check execution behind a feature flag with dual-run validation.

### Completed so far

- Phase 4 screen workbench implemented and screenshot-QA approved.
- `SchedulesWorkbench` replaces the legacy `SchedulesScreen` route.
- `Tests/UI/test_schedules_workbench.py` covers pane rendering, task selection, detail/inspector metadata, status badges, empty states, delete confirmation flow, service-error handling, and cron humanization.
- Full UI regression run: **122 passed, 1 skipped**.
- ADR-019 created for the watchlist scheduler migration.
