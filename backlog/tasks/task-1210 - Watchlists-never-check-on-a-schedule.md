---
id: TASK-1210
title: Watchlists never check on a schedule - promote the unified handler out of shadow mode
status: To Do
assignee: []
created_date: '2026-07-27 22:15'
labels:
  - watchlists
  - scheduling
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Automatic watchlist checking is unimplemented end to end, so a watchlist only ever updates when a
user presses "Check now".

The unified `WatchlistCheckHandler` ships disabled: `[scheduling] watchlist_checks_enabled` is
`false`, so `app.py` never constructs the handler, never registers a `watchlist_job` entry in the
scheduler's handler map, and passes `watchlist_projection=None` to `SchedulerLoop`. When the flag
is enabled, `watchlist_checks_shadow` still ships `true`, which fetches feeds and then discards
the results — `record_check_result` is skipped and `URLMonitor` is constructed with
`persist_snapshots=False`.

ADR-019 designates the legacy `SubscriptionScheduler` as the execution authority that the flag
falls back to, but that scheduler is unreachable in the shipped app (its only construction site
serves a `SubscriptionWindow` class that no longer exists). So the documented rollback lever is
dead: setting the flag false leaves no executor at all rather than restoring the old one.

This blocks the scheduled-delivery half of the Watchlists briefing spec — there is no working
schedule for a briefing to attach to.

Audit: `Docs/superpowers/research/2026-07-27-briefing-subsystem-revive-or-retire.md`
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A watchlist source with a check frequency is executed automatically by SchedulerLoop under default configuration, with no config.toml edit required
- [ ] #2 Results of an automatic check are persisted to Subscriptions_DB - record_check_result runs and URLMonitor persists snapshots
- [ ] #3 The Watchlists UI exposes a check frequency and round-trips it to Subscriptions_DB
- [ ] #4 Shadow mode remains available as an explicit opt-in for diagnostics, and its discard-results behaviour is documented where the flag is defined
- [ ] #5 A regression test asserts a due watchlist task dispatches to the handler and mutates the DB under default configuration
- [ ] #6 A test covers the case where neither scheduling flag is present in config.toml
- [ ] #7 ADR-019 is amended or superseded to record that the dual-run fallback it describes does not exist
<!-- AC:END -->
