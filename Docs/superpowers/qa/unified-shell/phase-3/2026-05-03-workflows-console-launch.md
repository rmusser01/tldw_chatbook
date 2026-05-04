# Workflows Destination Console Launch

Date: 2026-05-03
Task: `TASK-3.10`
Branch: `codex/unified-shell-phase3-workflows-live-work`
Base: `origin/dev` at `0fbbe48e`

## Purpose

Continue Phase 3 by making the Workflows destination expose a real Console launch action when the Home active-work adapter has actionable workflow-run context.

## What Changed

- Added a Workflows destination Console launch producer using the existing Home active-work adapter seam.
- Kept the Workflows Console launch action disabled with explicit recovery copy when no active workflow run exists.
- Enabled `workflows-launch-in-console` when an active `Workflows` item has Console context.
- Routed Workflows Console launch through `open_active_home_item_in_console` so Home and destination surfaces share the same launch path.
- Updated Console source readiness so Workflows is shown as connected.

## Functional QA Evidence

Focused checks were run against the Workflows destination and Console handoff harness.

- Baseline before the slice: `python -m pytest -q Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_destination_shells.py --tb=short`
- Baseline result: `73 passed, 3 warnings in 22.46s`.
- Red Workflows Console launch test: `python -m pytest -q Tests/UI/test_console_live_work_handoffs.py --tb=short`
- Red result: `8 failed, 35 passed, 1 warning in 14.92s`; failures were expected because Workflows still rendered an unavailable Console action, source readiness still marked Workflows as `Not wired`, adapter discovery was missing, and this evidence/task tracking did not exist.
- Focused Console handoff verification after implementation: `python -m pytest -q Tests/UI/test_console_live_work_handoffs.py --tb=short`
- Focused Console handoff result: `43 passed, 1 warning in 14.07s`.
- Broader Phase 3 regression check: `python -m pytest -q Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_destination_shells.py Tests/UI/test_home_screen.py Tests/Home/test_active_work_adapter.py Tests/UI/test_subscription_window_watchlists.py Tests/UI/test_screen_navigation.py Tests/UI/test_unified_shell_phase2_home_adapter.py --tb=short`
- Broader Phase 3 result: `174 passed, 8 warnings in 83.11s`.

Warning boundary: warnings are existing dependency/import warnings and are not Workflows Console-launch behavior failures.

## UX Result

- First-time users still see an honest unavailable state when no workflow run can be launched.
- Power users can jump directly from Workflows to Console when an active workflow-run context exists.
- The Workflows surface now matches the W+C and Schedules adapter-backed Console pattern instead of remaining a permanent false-negative disabled action.

## Residual Risk

- This slice wires the destination producer to adapter-provided context; it does not create a workflow-run backend snapshot adapter.
- Console still does not subscribe to real live event streams.
- Full Phase 3 still requires ACP, MCP, server Artifacts, and deeper live-work producers.
