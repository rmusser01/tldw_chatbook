# Home W+C Active-Work Console Launch

Date: 2026-05-03
Task: `TASK-3.3`
Branch: `codex/unified-shell-phase3-console-source-status`
Base: `origin/dev` at `19605f0f`

## Purpose

Make Home W+C active-work rows produce a real `ConsoleLiveWorkLaunch` so local watchlist runs can be inspected from the primary Console surface instead of stopping at an unavailable Console action.

## What Changed

- Local W+C watchlist run active-work rows now advertise Console availability when the Home adapter can resolve the visible run.
- `LocalNotificationHomeActiveWorkAdapter` handles `OPEN_IN_CONSOLE` for visible `local:watchlist_run:*` targets and returns a `HomeConsoleLaunch`.
- The launch includes W+C source, run title, current status, recovery copy, action label, and run metadata.
- App-level Home Console routing now preserves launch status, recovery, and action label when staging the Console status card.
- Unknown local watchlist run targets remain recoverable unavailable states.

## Functional QA Evidence

Focused checks were run against the Home adapter, app Console staging hook, and Phase 3 tracking evidence.

- Red test: `python -m pytest Tests/Home/test_active_work_adapter.py::test_local_notification_adapter_opens_local_watchlist_run_in_console Tests/UI/test_home_screen.py::test_app_console_hook_preserves_status_recovery_and_action_label Tests/UI/test_console_live_work_handoffs.py::test_phase3_home_wc_console_launch_tracking_evidence_links_task_and_roadmap -q`
- Red result: `3 failed`; failures covered unavailable W+C Console rows, missing HomeConsoleLaunch status/recovery/action metadata, and missing Phase 3.3 evidence.
- Green targeted result: `3 passed, 8 warnings in 5.65s`.
- Final focused Phase 3.3 suite: `88 passed, 8 warnings in 43.04s` for `Tests/Home/test_active_work_adapter.py`, `Tests/UI/test_home_screen.py`, `Tests/UI/test_console_live_work_handoffs.py`, `Tests/UI/test_screen_navigation.py`, and `Tests/UI/test_unified_shell_phase2_home_adapter.py`.

Warning boundary: warnings are existing dependency/import warnings and are not Home W+C Console launch failures.

## UX Result

- Home W+C active work now has a meaningful `Open in Console` path.
- Console receives the same status-card vocabulary introduced in `TASK-3.2`, with W+C-specific recovery guidance.
- This is the first source-specific live-work producer for the Console hub after the generic launch contract and status-card seam.

## Residual Risk

- This slice does not implement retry, pause, or resume for local watchlist runs from Console.
- This slice does not subscribe to live watchlist run event streams.
- Full Phase 3 still needs source producers for workflows, schedules, ACP, MCP, RAG, artifacts, and additional active-work sources.
