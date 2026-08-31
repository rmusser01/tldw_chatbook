---
id: TASK-26026
title: 'Scheduling: durable per-run execution ledger for reminders and briefings'
status: To Do
assignee: []
created_date: '2026-08-31 15:45'
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
- [ ] #1 Each reminder and briefing dispatch writes a durable run row recording start, finish, outcome and error where present
- [ ] #2 Run history is visible from the task detail surface, not only the latest outcome
- [ ] #3 Rows are pruned by a documented retention bound so the table cannot grow without limit
- [ ] #4 A run interrupted by application exit is reconciled on next start to a terminal state rather than left running - following the pattern at Subscriptions/startup_reconcile.py:164
- [ ] #5 The existing missed-fire accounting (missed_at, missed_count) continues to work and is not duplicated by the ledger
- [ ] #6 Server-scoped tasks are excluded: their run history remains server-authoritative per ADR-077
<!-- AC:END -->
