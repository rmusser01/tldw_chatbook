# Console W+C Action Routing

Date: 2026-05-03
Task: `TASK-3.4`
Branch: `codex/unified-shell-phase3-console-action-routing`
Base: `origin/dev` at `96f15eae`

## Purpose

Make the Console live-work action for W+C watchlist run launches complete a follow-through path into the existing W+C run detail surface instead of only rendering static action copy.

## What Changed

- `ConsoleLiveWorkStatusCardState` now exposes optional primary action metadata for supported launch payloads.
- W+C watchlist run launches with a `target_id` render a `console-live-work-primary-action` button.
- Clicking the Console live-work action stages `pending_subscription_initial_tab="watchlist-runs"` and `pending_subscription_watchlist_run_id` before navigating to W+C.
- Unsupported launch payloads remain non-actionable status cards with recovery copy.

## Functional QA Evidence

Focused checks were run against the Console status-card state, rendered Console button path, app-level W+C routing, unsupported fallback, and Phase 3.4 tracking evidence.

- Red test: `python -m pytest Tests/UI/test_console_live_work_handoffs.py::test_console_live_work_status_card_state_exposes_wc_primary_action Tests/UI/test_console_live_work_handoffs.py::test_console_live_work_status_card_state_keeps_unsupported_payloads_non_actionable Tests/UI/test_console_live_work_handoffs.py::test_app_console_live_work_primary_action_routes_wc_run_details Tests/UI/test_console_live_work_handoffs.py::test_console_wc_live_work_action_button_routes_run_details Tests/UI/test_console_live_work_handoffs.py::test_phase3_console_wc_action_tracking_evidence_links_task_and_roadmap -q`
- Red result: `5 failed`; failures covered missing primary action metadata, missing app routing helper, missing rendered button, and missing Phase 3.4 evidence.
- Green targeted result: `5 passed, 1 warning in 4.88s`.
- Final focused Phase 3.4 suite: `117 passed, 8 warnings in 50.15s` for `Tests/UI/test_console_live_work_handoffs.py`, `Tests/UI/test_home_screen.py`, `Tests/Home/test_active_work_adapter.py`, `Tests/UI/test_subscription_window_watchlists.py`, `Tests/UI/test_screen_navigation.py`, and `Tests/UI/test_unified_shell_phase2_home_adapter.py`.

Warning boundary: warnings are existing dependency/import warnings and are not Console W+C action routing failures.

## UX Result

- The Console status card now offers a real W+C follow-through action when the launch payload is actionable.
- Users can recover from a failed W+C active-work run by opening its W+C run details from Console.
- Unsupported sources still avoid false affordances until their service-specific routes exist.

## Residual Risk

- This slice does not add retry, pause, resume, or cancel controls from Console.
- This slice does not add live event subscriptions for watchlist runs.
- Workflow, schedule, ACP, MCP, RAG, artifact, and other source-specific Console actions remain intentionally unwired until their payload contracts exist.
