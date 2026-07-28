---
id: TASK-1210
title: Watchlists never check on a schedule - promote the unified handler out of shadow mode
status: In Progress
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
- [x] #1 A watchlist source with a check frequency is executed automatically by SchedulerLoop under default configuration, with no config.toml edit required
- [x] #2 Results of an automatic check are persisted to Subscriptions_DB - record_check_result runs and URLMonitor persists snapshots
- [x] #3 The Watchlists UI exposes a check frequency and round-trips it to Subscriptions_DB
- [x] #4 Shadow mode remains available as an explicit opt-in for diagnostics, and its discard-results behaviour is documented where the flag is defined
- [x] #5 A regression test asserts a due watchlist task dispatches to the handler and mutates the DB under default configuration
- [x] #6 A test covers the case where neither scheduling flag is present in config.toml
- [x] #7 ADR-019 is amended or superseded to record that the dual-run fallback it describes does not exist
<!-- AC:END -->

## Implementation Notes

Two one-word config changes and a UI field; the work was in proving the seam actually closes.

**Why the flags could not simply be "flipped when validation completes".** ADR-019 gates promotion
on dual-run parity metrics against the old scheduler. That gate is unsatisfiable — the old scheduler
is unreachable, so there is no second path to measure. The ADR is amended rather than satisfied:
`WatchlistCheckHandler` is promoted directly to sole executor, and the amendment records that the
rollback lever the ADR describes does not exist.

**Defaults are changed in two places on purpose.** The shipped TOML and `app.py`'s `get_cli_setting`
fallbacks are now consistent. They had drifted the same way, which is why nobody with an older
`config.toml` saw different behaviour from a new user and noticed something was wrong.

**Shadow mode has a second failure mode worth knowing.** It skips `record_check_result`, so
`last_checked` never advances, so `WatchlistProjection` keeps computing a `next_run_at` in the past
and the source is refetched on every queue reload — every ~30 minutes regardless of its cadence.
Left as documented behaviour (it is now diagnostics-only) rather than fixed, but it is a reason not
to leave the flag on against third-party feeds.

**The UI field needed no service change.** `LocalWatchlistsService._subscription_config_fields`
already allowed `check_frequency` through to `add_subscription`; only the form was missing it. It
shares a row with Tags because the Sources pane is 16 rows at 160x42 and a sixth full-height field
row puts Create/Cancel past the bottom edge — the same constraint the Type/Active row documents.

**Tests target the gap that let this ship.** Every component here already had isolated coverage and
all of it passed while the feature did nothing. `test_watchlist_scheduling_end_to_end.py` drives a
real `Subscriptions_DB` row through the projection, the queue and a real `SchedulerLoop`, and
asserts the result lands back in the database. Both persistence assertions are mutation-checked, as
is the overlay-clipping assertion.

**One live-verification note.** A tmux capture appeared to show the cadence dropdown opening with no
options, and the pre-existing toolbar Select looked identical — it read like a screen-wide defect.
`Screen._compositor.render_strips()` shows all four options painted correctly. The capture was a
harness artifact, the fourth on this programme; the compositor is the authority, not terminal art.

Modified: `tldw_chatbook/config.py`, `tldw_chatbook/app.py`,
`tldw_chatbook/UI/Watchlists_Modules/sources_pane.py`,
`tldw_chatbook/css/features/_watchlists.tcss` (+ regenerated bundle),
`backlog/decisions/019-watchlist-scheduler-migration.md`.
Tests: `Tests/Scheduling/test_watchlist_scheduling_end_to_end.py` (new),
`Tests/UI/test_watchlists_source_frequency_control.py` (new),
`Tests/Scheduling/test_config_flags.py`, `Tests/Watchlists/test_watchlists_sources_pane.py`,
`Tests/Subscriptions/test_local_watchlists_service.py`,
`Tests/UI/test_watchlists_source_create_form.py`.
