# Home Notification Review Routing

Date: 2026-05-03
Task: `TASK-4.5`
Branch: `codex/unified-shell-phase2-home-notification-review`
Base: `origin/dev` at `6ba7304f`

## Purpose

Make the Home unread-notification next-best action lead to the existing local notifications inbox instead of looping users back to Home or requiring them to remember where notifications are reviewed.

## What Changed

- Changed the `review_notifications` next-best action target route from `home` to `subscriptions`.
- Added a Home primary-action preparation hook that stages `pending_subscription_initial_tab = "notifications"` before navigation.
- Added a `SubscriptionWindow` initial-tab seam that consumes the pending notifications tab request, validates it against known tabs, and clears the one-shot request.
- Preserved notification count behavior and avoided creating active-work controls for generic notifications.

## Functional QA Evidence

Focused checks were run against pure Home dashboard state, mounted Home screen navigation, and the SubscriptionWindow tab context seam.

- Red test: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Home/test_dashboard_state.py::test_next_best_action_surfaces_notifications_after_live_work_blockers Tests/UI/test_home_screen.py::test_home_notification_primary_action_opens_notifications_inbox_context Tests/UI/test_subscription_window_watchlists.py::test_subscription_window_consumes_pending_notifications_initial_tab -q`
- Red result: `3 failed`; Home still routed notification review to `home`, and `SubscriptionWindow.initial_tab` did not exist.
- Green test: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Home/test_dashboard_state.py::test_next_best_action_surfaces_notifications_after_live_work_blockers Tests/UI/test_home_screen.py::test_home_notification_primary_action_opens_notifications_inbox_context Tests/UI/test_subscription_window_watchlists.py::test_subscription_window_consumes_pending_notifications_initial_tab -q`
- Green result: `3 passed, 8 warnings`
- Focused Phase 2 Home/subscription suite: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Home/test_dashboard_state.py Tests/Home/test_active_work_adapter.py Tests/UI/test_home_screen.py Tests/UI/test_subscription_window_watchlists.py Tests/UI/test_screen_navigation.py Tests/UI/test_unified_shell_phase2_home_adapter.py -q`
- Focused Phase 2 Home/subscription suite result: `91 passed, 8 warnings`

Warning boundary: warnings are existing dependency/import warnings and are not notification review routing failures.

## UX Result

- The Home "Review notifications" CTA now leads toward the real inbox workflow.
- Users do not need to know that the current inbox implementation lives under the subscriptions/W+C surface.
- The initial-tab request is one-shot, so later manual visits to Subscriptions still default to the normal subscriptions tab.

## Residual Risk

- This slice does not redesign the notifications inbox or move it into a dedicated top-level destination.
- The existing SubscriptionWindow notifications tab remains the current review surface.
- Full Phase 2 verification still requires real running-app QA with notification records produced by local jobs.
