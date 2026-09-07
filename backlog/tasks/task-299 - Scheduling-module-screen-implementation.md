---
id: TASK-299
title: Scheduling module + screen implementation
status: Done
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
- [x] #5 Watchlist jobs are visible in the workbench as read-only projections
- [x] #6 Watchlist checks execute under the unified scheduler (Phase 5)
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
- Task 5.4 redefinition (ADR-019): Phase 5's final step was redefined from removing the old `Subscriptions/scheduler.py` scheduler to deprecating it in place. The old scheduler is now retained and emits `DeprecationWarning`s so that dual-run validation can complete safely before the code is deleted in a follow-up release.

### Completed so far

- Phase 4 screen workbench implemented and screenshot-QA approved.
- `SchedulesWorkbench` replaces the legacy `SchedulesScreen` route.
- `Tests/UI/test_schedules_workbench.py` covers pane rendering, task selection, detail/inspector metadata, status badges, empty states, delete confirmation flow, service-error handling, and cron humanization.
- Full UI regression run: **122 passed, 1 skipped**.
- ADR-019 created for the watchlist scheduler migration.
- Phase 5 watchlist migration completed:
  - `WatchlistCheckHandler` implemented in `tldw_chatbook/Scheduling/scheduler/handlers/watchlist_check_handler.py`.
  - Feature flags `watchlist_checks_enabled` / `watchlist_checks_shadow` added to `[scheduling]` config.
  - `SchedulerLoop` and `PriorityQueue` optionally dispatch watchlist projections in shadow mode.
  - `SchedulingService.list_tasks()` merges reminders with `WatchlistProjection` results.
  - `SchedulesWorkbench` and `task_detail.py` render `ScheduledTask` read-only projections.
  - Old `Subscriptions/scheduler.py` and `textual_scheduler_worker.py` deprecated with `DeprecationWarning`s and ADR-019 linkage.
  - Lazy `__getattr__` in `Subscriptions/__init__.py` keeps package imports quiet while preserving backward compatibility.
- Final verification:
  - `Tests/Scheduling/`, `Tests/UI/test_schedules_workbench.py`, `Tests/UI/test_reminder_form.py`, `Tests/Subscriptions/test_scheduler_deprecation.py`: **194 passed**.
  - `ruff` clean on changed scheduling/UI files and `Subscriptions/__init__.py`.
  - `mypy` clean on changed scheduling files (`scheduling_service.py`, `schedules_workbench.py`, `task_detail.py`, `Subscriptions/__init__.py`); remaining errors in `app.py`, `config.py`, and legacy `Subscriptions/` scheduler files are pre-existing.
