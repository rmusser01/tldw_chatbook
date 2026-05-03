# Home Local Watchlist Run Snapshot

Date: 2026-05-03
Task: `TASK-4.6`
Branch: `codex/unified-shell-phase2-home-watchlist-runs`
Base: `origin/dev` at `52e4e89e`

## Purpose

Expose real local W+C watchlist run state on Home so queued, running, and failed local runs appear as active work instead of leaving Home as a notification-only dashboard.

## What Changed

- Added `LocalWatchlistsService.list_home_run_snapshot` as a synchronous Home-safe snapshot over recent local watchlist runs.
- Extended `LocalNotificationHomeActiveWorkAdapter` to accept the local watchlist service and map queued, running, paused, pending, and failed runs into `HomeActiveWorkItem` rows.
- Kept completed and cancelled local runs out of Home active-work rows.
- Prioritized failed active-work recovery through `review_failed_work` before generic active-work resume.
- Routed failed local watchlist work to the existing subscriptions `watchlist-runs` context.
- Wired `TldwCli` so the Home adapter receives both the local notification service and the local watchlist service.

## Functional QA Evidence

Focused checks were run against the local watchlist service, pure Home dashboard state, the Home adapter, mounted Home navigation, app service wiring, and Phase 2 tracking.

- Red test: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Subscriptions/test_local_watchlists_service.py Tests/Home/test_active_work_adapter.py Tests/Home/test_dashboard_state.py Tests/UI/test_home_screen.py Tests/UI/test_screen_navigation.py Tests/UI/test_unified_shell_phase2_home_adapter.py -q`
- Red result: `7 failed, 77 passed, 10 warnings`; failures covered the missing `list_home_run_snapshot`, missing `watchlist_service`, generic `resume_active_work` routing, missing `watchlist-runs` tab staging, and missing evidence links.
- Additional red-green self-review check: `Tests/Home/test_dashboard_state.py::test_failed_work_details_follow_failed_item_when_mixed_with_running_work` failed before detail selection followed the failed item, then passed after the selector fix.
- Green behavior test: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Subscriptions/test_local_watchlists_service.py Tests/Home/test_active_work_adapter.py Tests/Home/test_dashboard_state.py Tests/UI/test_home_screen.py Tests/UI/test_screen_navigation.py -q`
- Green behavior result: `73 passed, 8 warnings`
- Full focused Phase 2 test: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Subscriptions/test_local_watchlists_service.py Tests/Home/test_active_work_adapter.py Tests/Home/test_dashboard_state.py Tests/UI/test_home_screen.py Tests/UI/test_screen_navigation.py Tests/UI/test_unified_shell_phase2_home_adapter.py -q`
- Full focused Phase 2 result: `85 passed, 8 warnings`

Warning boundary: warnings are existing dependency/import warnings and are not Home watchlist-run failures.

## UX Result

- Home can now show concrete local W+C work such as "Daily security feed [failed] via W+C".
- Failed local watchlist work no longer looks like generic live chat work; the next-best action opens the W+C runs context for diagnosis.
- Completed local runs do not create stale active-work affordances.
- Retry remains behind the Home adapter boundary until a safe run-specific retry operation is implemented.

## Residual Risk

- This slice does not implement retry, pause, or resume for local watchlist runs; unavailable adapter messaging still handles those controls.
- The detail target uses the existing subscriptions/W+C surface rather than a dedicated run-detail route.
- Full Phase 2 verification still requires a running-app QA walkthrough with persisted local watchlist runs created from the UI.
