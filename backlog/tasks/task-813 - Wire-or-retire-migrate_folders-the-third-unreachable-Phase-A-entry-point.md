---
id: TASK-813
title: >-
  Wire or retire migrate_folders — the third unreachable Phase A entry point
status: To Do
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
- [ ] #1 A decision is recorded — wire it, or delete it — with the reasoning stated in the task notes
- [ ] #2 If wired: it runs once per database, off the UI thread, and never blocks startup
- [ ] #3 If wired: a database with hand-seeded `folder` values ends up with one watchlist per distinct folder, verified by a test
- [ ] #4 If deleted: `migrate_folders`, its tests, and the `watchlist_migration_state` marker table are removed together, and the tree's Unassigned root is confirmed to cover the same need
- [ ] #5 Either way, no method on `WatchlistBundleService` is left without a production caller
<!-- AC:END -->
