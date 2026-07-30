---
id: TASK-1383
title: Scheduled watchlist checks never create a run row, so the Runs pane cannot see them
status: To Do
assignee: []
created_date: '2026-07-30 02:01'
labels:
  - watchlists
  - scheduling
  - observability
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while implementing TASK-1362 (Task 3 review), whose §4 dispositions (`changed` /
`unchanged` / `withheld_below_threshold` / `baseline_stored`) were built to make silent checks
legible. They cannot reach the one path that is most invisible today.

`Scheduling/scheduler/handlers/watchlist_check_handler.py` (the scheduled-check path; see its
`:98-121`) calls `URLMonitor.check_url`, unpacks and **drops** the disposition it returns (a
deliberate TASK-1362 Task 3 decision — the sink it writes to has no field for one), then sinks the
result exclusively into `SubscriptionsDB.record_check_result` (`DB/Subscriptions_DB.py:1266`).
That method only ever writes `subscriptions.last_checked`/`last_error`/`consecutive_failures`,
`subscription_items`, and `subscription_stats` — a fixed-column daily aggregate
(`update_subscription_stats`, `DB/Subscriptions_DB.py:1562`) whose only reader,
`get_subscription_health` (`DB/Subscriptions_DB.py:1598`), has **zero callers anywhere outside
`Subscriptions_DB.py` itself** (verified by repo-wide grep).

The Watchlists Runs pane reads exclusively from `local_watchlist_runs`
(`UI/Screens/watchlists_collections_screen.py:2732` -> `LocalWatchlistsService.list_runs` ->
`Subscriptions/local_watchlists_service.py:487-490`), and the only code that ever inserts a row
into that table is `LocalWatchlistsService.launch_run`
(`Subscriptions/local_watchlists_service.py:317`, `INSERT INTO local_watchlist_runs` at `:329`) —
the manual "Check Now" / "Preview" path.

**Consequence:** a source that is only ever checked by the scheduler — the normal, unattended case
the feature exists for — never produces a row the Runs pane can show, with or without dispositions.
Its checks are not merely under-explained, they are entirely absent from the one screen built to
show what a check did. This is the same failure class as TASK-1210 (the scheduler silently doing
nothing) and TASK-1212 (the scheduler silently dropping unhandled task types): the machinery runs,
and there is nowhere a user can look to see that it ran.

Also folded in (Task 3 review, Minor 2): the scheduled-path end-to-end tests
(`Tests/Scheduling/test_watchlist_scheduling_end_to_end.py:74,99,127`) construct the handler with
`url_monitor=AsyncMock()`, so no test in that suite exercises `URLMonitor.check_url`'s real
contract (its return shape, its disposition, its snapshot writes) through the scheduled caller —
only through the manual path's tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A watchlist source checked only via the scheduler produces a run record the Runs pane displays, including its check disposition(s)
- [ ] #2 At least one scheduled-path test drives the real `URLMonitor` (not a mock), covering its actual `check_url` contract
- [ ] #3 A deliberately broken disposition on the scheduled path (e.g. dropped or miscounted) fails a test, proving the coverage discriminates it
<!-- AC:END -->
