---
id: TASK-15770
title: 'Watchlists: unread-badge count still scans instead of using the effective_date index'
status: Done
assignee:
  - '@claude'
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

- [x] `get_unread_items_count_since` uses the `effective_date` indexed
      column instead of recomputing the inline `COALESCE(datetime(...))`
      expression
- [x] `EXPLAIN QUERY PLAN` for the rewritten query shows the index is used
      (no full-table scan)
- [x] The returned count is identical to today's for both legacy rows
      (pre-migration, backfilled) and newly-inserted rows (parity test,
      mirroring task-15464's own ordering-parity test)
- [x] The Today badge's displayed count is unchanged before/after (existing
      Watchlists UI tests stay green)

## Implementation Plan

1. Re-locate at HEAD (`a58bdc126`): confirm `get_unread_items_count_since`
   still carries the inline `COALESCE(datetime(published_date),
   datetime(created_at))` predicate and that task-15464's `effective_date`
   generated column + `idx_subscription_items_effective_date` index exist.
2. BEFORE evidence: EXPLAIN QUERY PLAN probe on a REAL seeded SubscriptionsDB
   (5 subscriptions, 1k/10k items across ~200 days, mixed statuses,
   NULL/garbage published_date) capturing the full-table `SCAN` for the
   shipped shape, plus count + timing for both shapes. (Done — the shipped
   shape scans even on SQLite 3.49.1; the planner does NOT rewrite the
   inline expression to the generated column.)
3. Born-red plan-pin test: capture the SQL the method ACTUALLY executes via
   `set_trace_callback`, EXPLAIN it, and pin on the INDEX NAME appearing
   (per the alias-trap lesson: pin on the index name, not merely on the
   absence of the word SCAN). Red against the scanning shape.
4. Count-parity test mirroring task-15464's ordering-parity test: legacy
   pre-migration rows (hand-built schema, backfilled by opening through
   `SubscriptionsDB`) + rows inserted through `persist_subscription_item`,
   with at/before/after-floor rows, NULL and garbage `published_date`,
   timezone-offset dates that cross the floor only after normalization, and
   non-`new` statuses. Expected value = the OLD inline expression
   independently recomputed against the live table, not a hand-written
   number.
5. Fix the query shape only (no schema change — the index already exists):
   `effective_date >= datetime(?)` replacing the inline COALESCE.
6. AFTER evidence: re-run the probe (SEARCH ... USING INDEX
   idx_subscription_items_effective_date), identical counts, timings.
7. Gates: the SubscriptionsDB watchlists test file + Watchlists UI badge
   tests; ruff check + format on touched files; baseline any unexplained
   failures against origin/dev.

## Implementation Notes

Query-shape fix only -- no schema change, no migration. The predicate in
`SubscriptionsDB.get_unread_items_count_since` now reads the `effective_date`
generated column (`effective_date >= datetime(?)`) instead of respelling the
inline `COALESCE(datetime(published_date), datetime(created_at))`, letting
task-15464's existing `idx_subscription_items_effective_date` serve the floor
as an index range. The column IS the old expression (GENERATED ALWAYS AS the
same COALESCE), so the count is provably unchanged.

**Why the rewrite was needed at all:** probe-verified on the shipped SQLite
3.49.1 that the planner does NOT rewrite a byte-identical inline expression to
the generated column for index use -- the old shape full-scans even though the
index existed. EXPLAIN QUERY PLAN evidence from a real seeded SubscriptionsDB
(5 subscriptions, 1k/10k items across ~200 days, mixed statuses, NULL/garbage
`published_date`):

- BEFORE: `SCAN subscription_items`
- AFTER: `SEARCH subscription_items USING INDEX
  idx_subscription_items_effective_date (effective_date>?)`
- Counts identical at every scale/floor probed (129/25 @ 1k; 1286/257 @ 10k).
- Timing @ 10k rows: broad floor 2.85 -> 1.11 ms; narrow today-like floor
  2.93 -> 0.21 ms (~14x). @ 1k: 0.24 -> 0.02 ms narrow.

**Tests** (Tests/DB/test_subscriptions_db_watchlists.py):
- `test_get_unread_items_count_since_is_index_served` -- born-red plan pin:
  captures the SQL the method ACTUALLY executes via `set_trace_callback`
  (so the pin cannot drift from production), EXPLAINs it, and pins on the
  INDEX NAME appearing (alias-trap lesson), with no-SCAN as secondary guard.
  Red against the pre-task shape; green after.
- `test_unread_count_parity_old_inline_expression_vs_rewritten_query` --
  mirrors task-15464's ordering-parity test: legacy pre-migration rows
  backfilled through the ALTER path + rows via `persist_subscription_item`;
  at/before/after-floor rows, inclusive-boundary row, NULL and garbage
  `published_date`, timezone-offset dates that cross the floor only after
  `datetime()` normalization, read/unread transitions, a hard-deleted row;
  expected value = the OLD inline expression independently recomputed per
  floor (5 floors incl. offset-bearing), with a nonzero guard against a
  vacuous both-zero pass.

**Gates:** Tests/DB/test_subscriptions_db_watchlists.py 65/65 passed;
Tests/Watchlists/ + Tests/UI/test_watchlists_rail_counts_and_scope.py +
Tests/UI/test_watchlists_check_now_failure.py + Tests/Subscriptions/ =
1496 passed, 1 pre-existing skip (briefing-voices slice gate); full
`--collect-only` sweep 48,156 collected, no errors. ruff check clean on both
touched files; ruff format applied to the appended test range only (the rest
of the test file predates format enforcement and was left untouched to avoid
unrelated churn).

**Files:** `tldw_chatbook/DB/Subscriptions_DB.py`
(`get_unread_items_count_since`), `Tests/DB/test_subscriptions_db_watchlists.py`.
