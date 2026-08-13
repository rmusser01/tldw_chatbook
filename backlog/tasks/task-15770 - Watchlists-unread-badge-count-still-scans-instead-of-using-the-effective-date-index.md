---
id: TASK-15770
title: 'Watchlists: unread-badge count still scans instead of using the effective_date index'
status: To Do
assignee: []
created_date: '2026-08-13 12:31'
labels:
  - perf
  - watchlists
  - db
priority: low
---

## Description

Cheap follow-up named explicitly in task-15464's "Deliberate scope
boundaries" section (input-latency burn-down). Task-15464 added
`SubscriptionsDB`'s `effective_date` generated column and index (replacing
the unindexable `COALESCE(datetime(published_date), datetime(created_at))`
ordering used by the items list) but deliberately left
`get_unread_items_count_since` — a COUNT-only badge query — on the old
inline `COALESCE(...)` expression, since it was out of that task's "items
list queries" AC scope. It is still correct (the `effective_date` column's
existence doesn't change what the inline expression computes), just not
using the index task-15464 built.

## Acceptance Criteria

- [ ] `get_unread_items_count_since` uses the `effective_date` indexed
      column instead of recomputing the inline `COALESCE(datetime(...))`
      expression
- [ ] `EXPLAIN QUERY PLAN` for the rewritten query shows the index is used
      (no full-table scan)
- [ ] The returned count is identical to today's for both legacy rows
      (pre-migration, backfilled) and newly-inserted rows (parity test,
      mirroring task-15464's own ordering-parity test)
- [ ] The Today badge's displayed count is unchanged before/after (existing
      Watchlists UI tests stay green)
