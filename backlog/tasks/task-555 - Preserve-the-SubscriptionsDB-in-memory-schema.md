---
id: TASK-555
title: Preserve the SubscriptionsDB in-memory schema
status: Done
assignee: []
created_date: '2026-07-25 17:15'
updated_date: '2026-07-25 17:23'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the documented in-memory SubscriptionsDB mode retain its initialized schema so watchlist previews and test/runtime projections do not reconnect to an empty SQLite database.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 SubscriptionsDB(':memory:') remains queryable after initialization
- [x] #2 File-backed schema initialization connections still close promptly
- [x] #3 Closing an in-memory SubscriptionsDB releases its retained connection
- [x] #4 The subscriptions and affected Schedules handoff tests pass
- [x] #5 Task notes record RED evidence ADR applicability and verification
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused RED test proving an in-memory `SubscriptionsDB` loses its schema after initialization.
2. Retain the initialization connection only for in-memory instances while preserving file-backed close behavior and error cleanup.
3. Run Subscriptions tests and the affected Schedules handoff test.
4. Review and document the storage invariant.

ADR required: no; existing ADR applies
ADR path: backlog/decisions/019-watchlist-scheduler-migration.md
Reason: This is a routine bug fix restoring the existing documented `SubscriptionsDB` contract used by the ADR-019 watchlist projection; it introduces no new storage ownership or schema decision.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Restored the documented SubscriptionsDB(:memory:) contract by retaining the schema-initialization connection only for in-memory instances. File-backed initialization still closes its temporary handle, close() releases the retained in-memory handle, and initialization exceptions close the handle before propagating. Focused RED evidence was sqlite3.OperationalError: no such table: subscriptions from an immediate post-initialization query. Added regressions for retained schema, normal close, file-backed close, and failure cleanup. Final verification: 65/65 Subscriptions tests pass; Subscriptions plus screen navigation pass 122/122; screen navigation plus SchedulesWorkbench pass 93/93; Console live-work handoffs pass 47/47; Ruff, formatter, and diff checks pass. ADR required: no new ADR; backlog/decisions/019-watchlist-scheduler-migration.md already governs the watchlist projection and this change restores its existing storage contract. Modified: tldw_chatbook/DB/Subscriptions_DB.py, Tests/Subscriptions/test_subscriptions_smoke.py, and this task file.
<!-- SECTION:NOTES:END -->
