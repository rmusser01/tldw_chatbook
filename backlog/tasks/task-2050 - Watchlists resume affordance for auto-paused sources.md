---
id: TASK-2050
title: Watchlists resume affordance for auto-paused sources
status: Done
assignee: []
created_date: '2026-08-02'
labels:
  - watchlists
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix wave for the task-1410 review (Finding #1) gave an auto-paused source its first real recourse:
`SubscriptionsDB.record_check_result`'s success branch now writes `is_paused = 0` alongside its
existing counter reset, so a check that succeeds resumes a paused source. Combined with
`launch_run`/`execute_run` never having a paused guard, a **manual re-check** of a paused source
runs, and if it succeeds the source resumes.

That recourse exists entirely at the data layer. There is still no explicit "Resume" / un-pause
action anywhere in the watchlists UI, and nothing in the UI distinguishes an auto-paused source from
one that is merely inactive or healthy. `SubscriptionsDB.update_subscription(is_paused=0)` and
`reset_subscription_errors` both exist as un-pause writes, but grep confirms neither has a caller
outside the DB layer itself — there is no UI (or service-layer) path that invokes either one.

Net effect: a source that auto-pauses after repeated failures is visible only as a silently
stalled feed. A user who does not already know to trigger a manual re-check (and does not know
that succeeding is what un-pauses it) has no way to tell the source is paused, let alone resume it,
without editing config directly or reverse-engineering the recourse above.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 A paused source (`is_paused = 1`) is visibly marked as paused somewhere in the watchlists UI, distinguishable from an inactive or healthy source
- [x] #2 The user can resume a paused source with a single UI action, without editing config or manually crafting a re-check
- [x] #3 That resume action clears `is_paused` and resets the failure counters (`error_count`, `consecutive_failures`, `last_error`) via an existing or new service-layer call, not a direct DB write from the UI layer
<!-- AC:END -->

## Implementation Notes

**AC#1.** `normalize_local_subscription_row` (`watchlist_normalizers.py`) now stamps a `paused`
boolean and gives `status_summary` a `paused > error > inactive > active` precedence. **Decision:
paused wins over error**, even though a source auto-paused by task-1410 always still carries the
`last_error` that caused the pause. Rationale: an error-first headline ("error (10)") is
indistinguishable from a source that is merely having a bad day but is *still being retried on
schedule* — the one fact a paused source's status needs to lead with is that it has stopped being
retried and needs an explicit Resume. This is a real trade-off, not a free win: today neither
`last_error` nor `error_count` is surfaced anywhere else in the watchlists UI for a *source* (only a
*run's* own `error_count` renders, in the Runs pane), so for the window a source is both paused and
carrying the error that caused it, this precedence trades away the only place that error text was
visible at all. The underlying `last_error`/`error_count` columns are untouched either way and
remain available to a future source-detail affordance. `normalize_server_watchlist_source` always
stamps `paused: False` — the server watchlist source model has no auto-pause concept yet.
`SourcesPane.source_status_text`/`_source_row_cells` already render `status_summary` generically and
apply no per-status color anywhere in this pane (only a selection-highlight style) — confirmed no
existing status-coloring mechanism to mirror, so none was added.

**AC#2/#3.** `InspectorPane` gains a `Resume` button in `#inspector-actions`, rendered only when the
selected entity is `entity_kind == "subscription"` and `paused` is truthy
(`InspectorPane._is_paused_subscription`) — never for a server-backed `watchlist_source` even if its
`paused` flag were somehow set. Pressing it posts `ResumeSourceRequested`, handled by
`WatchlistsCollectionsScreen.handle_resume_source_requested` → `_resume_source`, which calls
`LocalWatchlistsService.resume_source(source_id)` directly (local-only, the same reason
`_open_snapshot_view` bypasses `WatchlistsBackendController` for `url_snapshots` — there is no
server-side pause concept for the controller to route to). `resume_source` delegates to
`SubscriptionsDB.reset_subscription_errors`, which already performed exactly this reset for
`record_check_result`'s success branch (task-1410) and had zero callers outside the DB layer; this
task gives it its first external caller. On success: a `markup=False` toast ("Resumed <name>. It
will be checked on its normal schedule.") and a reload via the existing
`_load_sources_preserving_selection` (same reload Check now uses), which refreshes both the Sources
table's Status column and the Inspector's own `selected_entity` — so the Resume button disappears
once the source is actually resumed. Failures are caught, logged with `.opt(exception=True)` (type
only), and reported with an honest error toast.

**Consistency guard.** `resume_source` is a harmless no-op on an already-healthy source: the
underlying `UPDATE` zeroes counters that are already zero and clears an already-clear pause flag. The
UI never offers the button for a non-paused source; the method itself does not need that guard to
stay correct (pinned by `test_resume_on_a_source_that_is_not_paused_is_a_harmless_no_op`).

**Tests.** `Tests/Subscriptions/test_watchlist_normalizers.py` (paused/precedence/regression/server
cases), `Tests/UI/test_watchlists_inspector.py` (button visibility gate across paused/healthy/server/
non-source entities, a real button press posting `ResumeSourceRequested`, and a full real-DB
end-to-end press → `SubscriptionsDB` row → toast → reload, plus the no-op case). Mutation-verified:
dropping the `paused` emission reddened both the normalizer tests and the real-path Inspector
end-to-end test; making the handler skip `service.resume_source(...)` reddened the end-to-end DB
assertions. `Tests/DB/test_subscriptions_db.py`'s and
`Tests/Subscriptions/test_local_watchlists_service.py`'s task-1410 pause/resume tests are untouched
and stay green — `reset_subscription_errors` itself was not modified.

**Files touched:** `tldw_chatbook/Subscriptions/watchlist_normalizers.py`,
`tldw_chatbook/Subscriptions/local_watchlists_service.py`,
`tldw_chatbook/UI/Watchlists_Modules/inspector_pane.py`,
`tldw_chatbook/UI/Screens/watchlists_collections_screen.py`,
`Tests/Subscriptions/test_watchlist_normalizers.py`, `Tests/UI/test_watchlists_inspector.py`.

## Review fix wave (2026-08-03)

The branch review confirmed the predicted filter regression: with `paused` winning the
status_summary precedence, an auto-paused source (which always carries the error that caused the
pause) vanished from the Sources pane's "Error" filter bucket — the triage view for broken feeds
silently missing the most-broken ones. Fixed: the Error bucket now includes `paused` (commented at
the branch), and a dedicated "Paused" option was added to `_STATUS_OPTIONS` (the matcher's existing
fallthrough handles it). Pinned by `test_paused_sources_stay_in_the_error_bucket_and_get_their_own`,
mutation-verified (dropping the paused arm reds it).
