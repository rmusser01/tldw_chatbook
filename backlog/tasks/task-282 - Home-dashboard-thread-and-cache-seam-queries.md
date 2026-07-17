---
id: TASK-282
title: Home: thread + cache dashboard seam queries; targeted rail/canvas updates
status: In Progress
assignee: []
created_date: '2026-07-16 14:30'
labels: [performance, home]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
home_screen._build_dashboard_input runs 3 synchronous DB/repository queries (watchlist snapshot, notification queue limit=100, server-event feed) on the UI thread at every compose, triage sync, and rail click with no cross-visit cache — while the sibling _home_content_seam_call already uses asyncio.to_thread correctly. HomeRail/HomeCanvas.sync_state also always recompose. Full analysis with measurements: Docs/Design/2026-07-16-performance-audit.md (§P1 B3).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Dashboard seam queries run off the event loop with a short-TTL cache; Home still reflects fresh data per its seam contract
- [x] #2 Selection/count-only changes patch targeted widgets instead of recomposing rail/canvas
- [x] #3 Existing Home triage tests green
<!-- AC:END -->

## Implementation Plan

1. Re-verify the audit's line anchors against current code -- `home_screen._build_dashboard_input` itself no longer inlines the 3 queries; they live one level down in `LocalNotificationHomeActiveWorkAdapter.build_dashboard_input` (watchlist snapshot, notification count, server-event feed), called synchronously from `_build_dashboard_input` at every compose/triage-sync/rail-click.
2. Add a short-TTL (a few seconds) cache to the adapter itself (it lives on the app, not the per-visit HomeScreen, so the cache is cross-visit by construction -- the audit's "no cross-visit cache" complaint).
3. Add an async `refresh_active_work_cache_async` on the adapter that runs the 3 queries via `asyncio.to_thread`, gated off-thread only when the backing store is not a per-connection `:memory:` DB (mirrors `_home_content_seam_call`'s ChaChaNotes guard, but for `ClientNotificationsDB`, which caches a single non-thread-local sqlite connection for `:memory:` paths).
4. Wire a HomeScreen on-mount worker to call the adapter's async refresh, and an invalidation hook into `TldwCli._handle_home_control_action` (approve/reject/pause/resume/retry) so triage actions don't read a stale cache.
5. Convert `HomeRail.sync_state`/`HomeCanvas.sync_state` to patch targeted widgets (row selection marker/class, details text, canvas lines) when the structural shape is unchanged, falling back to `refresh(recompose=True)` for any structural difference.
6. Add new tests; run the existing Home suites to confirm no regressions.

## Implementation Notes

**Threading + caching** (`tldw_chatbook/Home/active_work_adapter.py`): `LocalNotificationHomeActiveWorkAdapter` gained a lock-protected `_active_work_cache` (dict of `runs`/`notification_count`/`server_event_fields`) with a 3s TTL. `build_dashboard_input` stays fully synchronous (Home's `compose_content` is a plain generator and cannot await) but now reads the cache instead of always querying; a cold/stale cache still computes inline as a safety net. `refresh_active_work_cache_async` (new) is the actual off-loop path: it runs `_compute_active_work_fields` via `asyncio.to_thread` unless `_active_work_seams_are_memory_backed()` is true (checks `notification_service.store.is_memory_db`, which is also `server_event_service.local_service.store` in this app's wiring), in which case it stays inline to avoid a `sqlite3.ProgrammingError` from `ClientNotificationsDB`'s non-thread-local `:memory:` connection cache. `invalidate_active_work_cache()` is a new public hook; `TldwCli._handle_home_control_action` (app.py) calls it (via `getattr`, defensive against the honest-unavailable adapter and test doubles) after approve/reject/pause/resume/retry.

`HomeScreen.on_mount` (`tldw_chatbook/UI/Screens/home_screen.py`) now also starts `_refresh_home_active_work_cache` (`@work(exclusive=True, group="home-active-work-cache")`), which awaits the adapter's async refresh (via `inspect.iscoroutinefunction` — silently skipped for adapters/test doubles that don't implement it) and re-syncs triage in place when mounted.

**Targeted rail/canvas patches**: `HomeRail.sync_state` now compares the new/previous `triage.sections` (deep dataclass equality) and `preferences`; when both are unchanged, it patches only the previously/newly selected row's Button label+class and the `#home-details-body` text instead of recomposing. Any row added/removed/relabelled, or a preference change, still triggers a full `refresh(recompose=True)`. `HomeCanvas.sync_state` similarly compares `title`/`actions`/`next_action`/`next_action_is_canvas`/`primary_control_id`; when all match but `lines` differs (e.g. the idle canvas's content-counts line updating from a background refresh), it patches only `#home-canvas-lines`; an identical `canvas` state is a full no-op (no `refresh()` call at all). Both patch branches wrap in `try/except` and fall back to a full recompose on any failure, per the task-280 lesson about widget-level recompose skipping breaking click targets.

**Files changed**: `tldw_chatbook/Home/active_work_adapter.py`, `tldw_chatbook/UI/Screens/home_screen.py`, `tldw_chatbook/app.py`, `tldw_chatbook/Widgets/Home/home_rail.py`, `tldw_chatbook/Widgets/Home/home_canvas.py`. New test file: `Tests/UI/test_home_dashboard_seams.py` (adapter cache/threading unit tests, widget-level targeted-patch-vs-recompose tests, HomeScreen on-mount wiring test). All 256 tests across `Tests/UI/test_home_screen.py`, `Tests/UI/test_home_triage_rail.py`, `Tests/Home/`, `Tests/UI/test_home_dashboard_seams.py`, and `Tests/UI/test_screen_navigation.py` pass.
