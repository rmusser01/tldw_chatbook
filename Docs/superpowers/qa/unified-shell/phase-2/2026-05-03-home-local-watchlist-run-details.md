# Home Local Watchlist Run Details

Date: 2026-05-03
Task: `TASK-4.7`
Branch: `codex/unified-shell-phase2-home-watchlist-details`
Base: `origin/dev` at `09ca0200`

## Purpose

Make Home `Open details` for local W+C watchlist run active-work items open the existing W+C runs surface with the relevant run selected and loaded, instead of stopping at an unavailable adapter message.

## What Changed

- Added handled `Open details` support in `LocalNotificationHomeActiveWorkAdapter` for visible `local:watchlist_run:*` items.
- Staged `pending_subscription_initial_tab = "watchlist-runs"` and `pending_subscription_watchlist_run_id` before navigating from Home details into subscriptions.
- Made `SubscriptionWindow` consume the pending watchlist run id, restore the selected runs row, and load the run detail payload.
- Kept local watchlist jobs honest as server-only while allowing local runs and alert rules to render through the existing scope service.

## Functional QA Evidence

Focused checks were run against the Home adapter, app detail navigation hook, SubscriptionWindow pending context, and local W+C runs rendering.

- Red test: `python -m pytest Tests/Home/test_active_work_adapter.py::test_local_notification_adapter_opens_local_watchlist_run_details Tests/UI/test_home_screen.py::test_app_detail_hook_stages_watchlist_runs_context_for_handled_watchlist_detail Tests/UI/test_subscription_window_watchlists.py::test_subscription_window_consumes_pending_watchlist_run_detail_context Tests/UI/test_subscription_window_watchlists.py::test_local_mode_shows_watchlist_control_plane_guidance Tests/UI/test_subscription_window_watchlists.py::test_local_mode_pending_watchlist_run_context_loads_run_detail -q`
- Red result: `5 failed`; failures covered unavailable adapter detail handling, missing Home subscription context staging, missing pending run consumption, and local W+C runs hidden behind local-only state.
- Green targeted result: `5 passed, 8 warnings`
- Full focused Phase 2.7 suite: `python -m pytest Tests/Home/test_active_work_adapter.py Tests/UI/test_home_screen.py Tests/UI/test_subscription_window_watchlists.py Tests/UI/test_screen_navigation.py Tests/UI/test_unified_shell_phase2_home_adapter.py -q`
- Full focused Phase 2.7 result: `93 passed, 8 warnings`

Warning boundary: warnings are existing dependency/import warnings and are not Home watchlist-run detail failures.

## UX Result

- Home `Open details` now takes the user to the W+C Runs tab for the selected local watchlist run.
- The selected run is restored in the runs list and its detail JSON is loaded for diagnosis.
- Local jobs remain honestly unavailable as a separate server-style control plane; local runs and alert rules are visible because local services already expose them.

## Residual Risk

- This slice does not implement Home retry/pause/resume for local watchlist runs.
- This slice does not redesign the W+C runs detail layout; it uses the existing JSON detail panel.
- Full Phase 2 verification still needs running-app QA with real persisted local runs created through the UI.
