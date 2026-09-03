---
id: TASK-26026
title: 'Scheduling: durable per-run execution ledger for reminders and briefings'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-31 15:45'
updated_date: '2026-09-01 20:27'
labels:
  - scheduling
  - ops
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reminder and briefing history is one overwritten row. Verified on origin/dev: mark_reminder_dispatched writes a single last_status and last_run_at pair on the task itself (Scheduling/db/scheduled_tasks_db.py:772-830), so run N-1 is unrecoverable; the audit table that exists, automation_audit_events (Scheduling/db/migrations/v0_to_v1.py:114), records definition CRUD, not executions. Watchlists already have a real ledger - local_watchlist_runs at DB/Subscriptions_DB.py:936-950, with orphan reconciliation at Subscriptions/startup_reconcile.py:164,195 - so this closes the gap for the two handlers that lack one, and that table is the shape to follow. NOTE: this corrects TASK-18936, which claimed execution-audit parity.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each reminder and briefing dispatch writes a durable run row recording start, finish, outcome and error where present
- [x] #2 Run history is visible from the task detail surface, not only the latest outcome
- [x] #3 Rows are pruned by a documented retention bound so the table cannot grow without limit
- [x] #4 A run interrupted by application exit is reconciled on next start to a terminal state rather than left running - following the pattern at Subscriptions/startup_reconcile.py:164
- [x] #5 The existing missed-fire accounting (missed_at, missed_count) continues to work and is not duplicated by the ledger
- [x] #6 Server-scoped tasks are excluded: their run history remains server-authoritative per ADR-077
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. v3_to_v4 migration: scheduled_task_runs table (mirrors local_watchlist_runs)\n2. DB methods: begin/finish_task_run, list_task_runs, prune_task_runs, fail_interrupted_task_runs\n3. Dispatch wiring: begin before handler, finish with outcome/error; ledger types {reminder, briefing_job}; server-scoped excluded (AC#6)\n4. run() startup: reconcile interrupted + prune (AC#3/#4)\n5. Task-detail Recent runs surface (AC#2); version-pin test updates; guide
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
New scheduled_task_runs ledger (ScheduledTasks DB v3->v4 migration, shape mirrors local_watchlist_runs) with one durable row per dispatch (task_id, task_type, status, started_at, finished_at, error_msg): begin_task_run (opens a 'running' row) / finish_task_run (terminal status+error) / list_task_runs (newest-first) / prune_task_runs (keeps newest DEFAULT_RUN_HISTORY_PER_TASK=50, AC#3) / fail_interrupted_task_runs (AC#4). Dispatch wiring in SchedulerLoop.dispatch_reminder opens the row before the handler and closes it with completed/timed_out/failed(+error) after, for _LEDGER_TASK_TYPES={reminder, briefing_job} only, excluding server-scoped owners (AC#6); every ledger call is fail-safe (never breaks dispatch) and guarded by hasattr so a stub db still works. AC#4 reconcile runs in run() BEFORE the poll loop starts — so no live run of THIS process can be wrongly failed (simpler than the watchlist row-boundary sweep, which runs alongside live work), plus a prune. AC#5: mark_reminder_dispatched (missed_at/missed_count) is untouched — the ledger is additive, not a replacement. AC#2: Task Detail gains a 'Recent runs' Static (pure format_run_history, newest-first with errors) fetched by the workbench via SchedulingService.db.list_task_runs (fail-safe). Migration NOT subject to the DB/ index-pin census (scanner scopes tldw_chatbook/DB only; scheduled_task_runs' index lives in Scheduling/db) — verified check_index_plan_pins green. Six version-pin tests updated 3->4 (fresh chain now reaches v4). Corrects TASK-18936's execution-audit-parity claim. 13 new tests; scheduler 388 + schedules UI 58 green.
<!-- SECTION:NOTES:END -->
