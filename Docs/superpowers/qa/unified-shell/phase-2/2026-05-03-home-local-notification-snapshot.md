# Home Local Notification Snapshot Adapter

Date: 2026-05-03
Task: `TASK-4.4`
Branch: `codex/unified-shell-phase2-home-local-snapshot-adapter`
Base: `origin/dev` at `252b7ba8`

## Purpose

Connect Home to the local notification queue so the dashboard can surface real unread notification state without misclassifying generic notifications as approvals, active runs, or controllable work items.

## What Changed

- Added `notification_count` to `HomeDashboardInput`.
- Added `LocalNotificationHomeActiveWorkAdapter`, which reads unread items from `ClientNotificationsService.list_queue`.
- Wired `TldwCli` to replace the unavailable Home adapter with the local notification snapshot adapter after local notification services initialize.
- Added Home dashboard and screen behavior so unread notifications appear in the Attention section and can become the next-best action after stronger live-work blockers.
- Preserved unavailable active-run controls until real active-run, schedule, workflow, or agent services expose safe adapter contracts.

## Functional QA Evidence

Focused checks were run against the pure Home dashboard state, the adapter contract, the app service wiring, and the mounted Textual Home screen harness.

- Red test: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Home/test_active_work_adapter.py Tests/Home/test_dashboard_state.py Tests/UI/test_home_screen.py Tests/UI/test_screen_navigation.py -q`
- Red result: collection failed because `LocalNotificationHomeActiveWorkAdapter` did not exist.
- Green test: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Home/test_active_work_adapter.py Tests/Home/test_dashboard_state.py Tests/UI/test_home_screen.py Tests/UI/test_screen_navigation.py -q`
- Green result: `58 passed, 8 warnings`
- Focused Phase 2 suite: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Home/test_dashboard_state.py Tests/Home/test_active_work_adapter.py Tests/UI/test_home_screen.py Tests/UI/test_screen_navigation.py Tests/UI/test_unified_shell_phase2_home_adapter.py -q`
- Focused Phase 2 suite result: `66 passed, 8 warnings`

Warning boundary: warnings are existing dependency/import warnings and are not Home notification snapshot behavior failures.

## UX Result

- Home now behaves more like a status and notification center instead of a static shell placeholder.
- Users can see unread local notifications from Home without needing to know the notification queue implementation.
- Generic notifications do not create approve, pause, retry, or Console controls, preventing false affordances.
- The next-best action order remains conservative: model readiness, approvals, failed schedules, and active work outrank notification review.

## Residual Risk

- This slice only covers local notification snapshot state.
- It does not add a notification detail drawer, batch mark-read actions, or category filters.
- Real active-run, schedule, workflow, and agent-service adapters still need implementation before Phase 2 can be verified end-to-end.
