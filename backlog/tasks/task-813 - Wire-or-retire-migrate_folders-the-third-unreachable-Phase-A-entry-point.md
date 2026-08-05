---
id: TASK-813
title: >-
  Wire or retire migrate_folders — the third unreachable Phase A entry point
status: Done
assignee: []
created_date: '2026-07-26 12:45'
labels:
  - watchlists
  - followup
  - tech-debt
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`WatchlistBundleService.migrate_folders()` has no production caller. Verified against the real user database: `watchlist_migration_state` is empty, so it has never run, and `watchlists` has 0 rows despite 3 existing subscriptions.

This is the **third** Phase A entry point that shipped complete, tested, and unreachable:

- `backfill_items_fts` — no caller (fixed in task-688, PR #929)
- `WatchlistBundleService` itself — never instantiated (fixed in Phase C task 1)
- `migrate_folders` — still unreachable

The pattern is worth naming: Phase A built a correct data layer and wired none of its entry points, and each one was only discovered when something downstream tried to use it.

Deciding what to do needs a judgement call, because the migration may be worthless by construction. Nothing in the codebase ever writes `subscriptions.folder` — `_subscription_config_fields` in `local_watchlists_service.py` allowlists ten fields and `folder` is not among them, and no caller passes `add_subscription(folder=…)`. So on any real database the migration can only ever produce a single `Unsorted` watchlist containing every source.

Meanwhile Phase C's tree already carries a permanent **Unassigned** root that shows exactly those sources. So wiring the migration might add nothing a user can see, while creating a watchlist they did not ask for.

Either outcome is defensible; what is not defensible is leaving a tested method that nothing calls.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A decision is recorded — wire it, or delete it — with the reasoning stated in the task notes
- [x] #2 If wired: it runs once per database, off the UI thread, and never blocks startup
- [x] #3 If wired: a database with hand-seeded `folder` values ends up with one watchlist per distinct folder, verified by a test
- [x] #4 If deleted: `migrate_folders`, its tests, and the `watchlist_migration_state` marker table are removed together, and the tree's Unassigned root is confirmed to cover the same need
- [x] #5 Either way, no method on `WatchlistBundleService` is left without a production caller
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Decision: deleted.**

The migration is a no-op by construction on any database this app can produce. `subscriptions.folder` is never written on a reachable path, so every source falls into the same bucket and the migration can only ever create one `Unsorted` watchlist containing everything — which is exactly what Phase C's permanent **Unassigned** tree root already shows. Wiring it would have created a watchlist the user never asked for, duplicating an existing view.

**Correction to this task's own description.** It claimed "nothing in the codebase ever writes `subscriptions.folder`". That is wrong: `Event_Handlers/subscription_events.py:275` does pass `folder=folder_input.value.strip() or None`. The conclusion still holds, for a stronger reason — that code is unreachable. `handle_add_subscription` has zero dispatchers (only message classes are imported from that module, by `textual_scheduler_worker.py`), and `#subscription-folder-input` is composed nowhere, so the `query_one` at `:213` would raise `NoMatches` before the write could happen.

**Removed:** `migrate_folders` and `MIGRATION_KEY` from `watchlist_bundle_service.py`; the `watchlist_migration_state` DDL from `Subscriptions_DB._initialize_schema`; its schema assertion in `Tests/DB/test_subscriptions_db_watchlists.py`; and the four `test_migrate_folders_*` tests. No `DROP TABLE` migration was written — existing databases keep an empty, unused table, which is harmless and strictly safer than dropping user data.

**AC #5, honestly.** Deleting `migrate_folders` did not by itself leave every method reachable: `create`, `rename`, `delete`, `add_source` and `remove_source` still have no production caller. Those are different in kind — orphaned pending UI rather than worthless — and Phase C deliberately shipped only the tree's read half. Filed as task-895 so the gap is tracked rather than rediscovered a phase later, which is the exact failure mode this task exists to name.
<!-- SECTION:NOTES:END -->
