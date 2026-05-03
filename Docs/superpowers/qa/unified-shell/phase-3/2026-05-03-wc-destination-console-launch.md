# W+C Destination Console Launch

Date: 2026-05-03
Task: `TASK-3.5`
Branch: `codex/unified-shell-phase3-wc-console-launch`
Base: `origin/dev` at `fa37fdb0`

## Purpose

Make the W+C destination produce a real Console follow action when an actionable W+C active-work run is available, while preserving an honest disabled state when no run context exists.

## What Changed

- `WatchlistsCollectionsScreen` now queries the existing Home active-work adapter for a latest Console-capable W+C run.
- `watchlists-follow-in-console` stays disabled with recovery copy when no active W+C run is available.
- When a Console-capable W+C run exists, `watchlists-follow-in-console` becomes enabled and identifies the latest run by title and status.
- Clicking the enabled W+C Console follow action routes through `open_active_home_item_in_console`, preserving the existing Home adapter and Console launch contracts.

## Functional QA Evidence

Focused checks were run against W+C destination fallback state, W+C destination enabled state, click routing, and Phase 3.5 tracking evidence.

- Red test: `python -m pytest Tests/UI/test_console_live_work_handoffs.py::test_watchlists_destination_keeps_console_follow_disabled_without_active_run Tests/UI/test_console_live_work_handoffs.py::test_watchlists_destination_routes_latest_active_run_to_console Tests/UI/test_console_live_work_handoffs.py::test_phase3_wc_destination_console_tracking_evidence_links_task_and_roadmap -q`
- Red result: `3 failed`; failures covered stale disabled copy, always-disabled Console follow, and missing Phase 3.5 evidence.
- Green targeted result: `3 passed, 1 warning in 4.70s`.
- Final focused Phase 3.5 suite: `154 passed, 8 warnings in 70.35s` for `Tests/UI/test_console_live_work_handoffs.py`, `Tests/UI/test_destination_shells.py`, `Tests/UI/test_home_screen.py`, `Tests/Home/test_active_work_adapter.py`, `Tests/UI/test_subscription_window_watchlists.py`, `Tests/UI/test_screen_navigation.py`, and `Tests/UI/test_unified_shell_phase2_home_adapter.py`.

Warning boundary: warnings are existing dependency/import warnings and are not W+C destination Console follow failures.

## UX Result

- W+C no longer presents Console follow as permanently unavailable when the app already has actionable W+C active-work context.
- The destination reuses the same active-work source of truth as Home, so users see a consistent latest W+C run across dashboard and destination surfaces.
- No active run still produces a clear, non-clickable recovery state rather than a false affordance.

## Residual Risk

- This slice follows the latest available W+C active-work item only; it does not add selection or filtering inside the W+C shell wrapper.
- Retry, pause, resume, and cancel controls for local watchlist runs remain deferred.
- Workflow, schedule, ACP, MCP, RAG, artifact, and other source-specific Console producers remain intentionally unwired until their service contracts exist.
