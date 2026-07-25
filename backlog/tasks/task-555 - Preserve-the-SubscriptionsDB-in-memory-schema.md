---
id: TASK-555
title: Preserve the SubscriptionsDB in-memory schema
status: In Progress
assignee: []
created_date: '2026-07-25 17:15'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the documented in-memory SubscriptionsDB mode retain its initialized schema so watchlist previews and test/runtime projections do not reconnect to an empty SQLite database.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 SubscriptionsDB(':memory:') remains queryable after initialization
- [ ] #2 File-backed schema initialization connections still close promptly
- [ ] #3 Closing an in-memory SubscriptionsDB releases its retained connection
- [ ] #4 The subscriptions and affected Schedules handoff tests pass
- [ ] #5 Task notes record RED evidence ADR applicability and verification
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
