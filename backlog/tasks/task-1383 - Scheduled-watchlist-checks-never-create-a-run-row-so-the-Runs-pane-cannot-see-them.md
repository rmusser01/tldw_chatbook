---
id: TASK-1383
title: Scheduled watchlist checks never create a run row, so the Runs pane cannot see them
status: Done
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
- [x] #1 A watchlist source checked only via the scheduler produces a run record the Runs pane displays, including its check disposition(s)
- [x] #2 At least one scheduled-path test drives the real `URLMonitor` (not a mock), covering its actual `check_url` contract
- [x] #3 A deliberately broken disposition on the scheduled path (e.g. dropped or miscounted) fails a test, proving the coverage discriminates it
<!-- AC:END -->

## Implementation Plan
<!-- SECTION:PLAN:BEGIN -->
1. Establish what the handler's fork diverged on, and whether the service path preserves its
   failure/auto-pause behaviour.
2. Route non-shadow checks through `LocalWatchlistsService.launch_run` + `execute_run`, deleting
   the fork's own dispatch and `record_check_result` block.
3. Keep shadow mode as a separate, documented no-mutation probe.
4. Cover with tests that drive the real `URLMonitor` with only the HTTP fetch faked.
5. Mutation-check that the new coverage discriminates each behaviour it claims.
<!-- SECTION:PLAN:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
**Unified rather than extended.** `WatchlistCheckHandler` was a parallel reimplementation of the
service's execution path. Non-shadow checks now call `launch_run(source_id=...)` +
`execute_run(run_id)`, which yields the run row, `stats_json` dispositions, per-URL baselines
(TASK-1361), filters and alerts for free. The handler's own dispatch and `record_check_result`
block are gone; it keeps the guards that are genuinely its job (task-id parse, missing
subscription, paused/inactive skip, metrics in `finally`).

**Two divergences fixed by deletion, not patching.** The fork's `_URL_TYPES = ("url", "url_list")`
omitted `sitemap`, so every scheduled sitemap source hit the "unknown subscription type" branch and
was never checked — while the `subscriptions.type` CHECK constraint had accepted `sitemap` all
along. And `url_list` passed the whole subscription to ONE `check_url` call, so a scheduled 50-URL
source checked a single URL. `EXECUTABLE_SOURCE_TYPES` (`local_watchlists_service.py`) is now the
one definition both the executor and the handler's guard read;
`test_executable_types_match_every_type_the_db_accepts` pins it against the schema's own CHECK
constraint, so "storable but not executable" cannot recur.

**Auto-pause: parity is exact, because there is none.** The task brief assumed the old path
auto-paused via `record_check_error`. It does not. Auto-pause's only implementation is the `if
error:` branch of `record_check_result` (`DB/Subscriptions_DB.py:1318-1341`), and that branch has
no caller — the sole production caller (`local_watchlists_service.py:448`) never passes `error`.
`record_check_error` (`:1391-1411`) bumps `consecutive_failures` but writes `is_paused = 1 if
should_pause else 0`, defaulting to `False`. The old handler called
`record_check_error(subscription_id, str(exc))`; the service reaches the *identical* call from
`record_run_failure` (`local_watchlists_service.py:509`). So the counter advances exactly as
before — and adding a compensating call in the handler's `except` would have double-bumped it.
The latent bug (nothing ever auto-pauses; every failure actively clears `is_paused`) is out of
scope here and filed as TASK-1410.

**Intended behaviour change.** Scheduled checks now also run filter and content-alert
*evaluation*, because that is part of `execute_run` and unifying onto it is the point — a
scheduled check and a manual one should not disagree about which items survive filtering or which
alert rules matched. No notification is *dispatched*: the handler's service is constructed with
`notification_dispatcher=None`, so evaluation results are recorded against the run and nothing is
sent. Wiring a dispatcher into the scheduled path is a separate decision, not a side effect of
this one.

**No double-write.** `execute_run` calls `record_check_result` itself, and it does not re-raise a
fetch failure — so the handler neither records a second time nor expects an exception for the
ordinary failure case. Its `except` now only sees faults *around* execution (`launch_run` raising,
or `execute_run` failing before its own `try`), and routes them to `record_run_failure` so the
just-inserted row cannot be orphaned at `queued` — the TASK-1090 shape, newly reachable from the
scheduler because the scheduler launches runs at all.

**Shadow mode** keeps the direct-monitor probe, isolated in `_check_in_shadow` and documented as
the deliberate fork: it cannot use the run seam because that seam exists to write. It stays
deliberately coarser (one URL for a `url_list`, no sitemap enumeration) — a diagnostic's fidelity,
safe only because nothing it returns is persisted.

**Tests.** `Tests/Scheduling/test_scheduled_watchlist_runs.py` (10 tests) drives the real
`URLMonitor` through the real handler with only `guarded_fetch_httpx_async` faked (AC#2); every
pre-existing scheduled-path test passed `url_monitor=AsyncMock()`. All same-thread asyncio, but
file-backed DBs under `tmp_path` so a later thread-hop fails loudly rather than finding an empty
schema. The handler's unit tests moved to the delegation contract; the end-to-end feed test stubs
at the service's `run_executor` seam so real persistence stays in the loop.

**Mutation checks** (AC#3), each restored afterwards:
- bare `check_url` + `record_check_result` -> 5 failed (both AC#1 tests and the `url_list` count);
- the old type tuples -> only the sitemap regression failed, 1 failed / 9 passed;
- shadow guard dropped -> 2 failed (both shadow tests);
- `_record_failure` no-op -> 3 failed;
- `record_run_failure`'s `record_check_error` removed -> 2 failed (both auto-pause tests).

`Tests/Scheduling/ Tests/Subscriptions/ Tests/Watchlists/`: **587 passed**, 0 failed.

Modified: `Scheduling/scheduler/handlers/watchlist_check_handler.py`,
`Subscriptions/local_watchlists_service.py`, `Tests/Scheduling/test_watchlist_check_handler.py`,
`Tests/Scheduling/test_watchlist_scheduling_end_to_end.py`. Added:
`Tests/Scheduling/test_scheduled_watchlist_runs.py`, `backlog/tasks/task-1410`.
<!-- SECTION:NOTES:END -->
