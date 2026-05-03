# Schedules Destination Console Launch

Date: 2026-05-03
Task: `TASK-3.7`
Branch: `codex/unified-shell-phase3-schedules-live-work`
Base: `origin/dev` at `bce5430e`

## Purpose

Continue Phase 3 by making the Schedules destination expose a real Console follow action when the Home active-work adapter has actionable schedule-run context.

## What Changed

- Added a Schedules destination Console follow producer using the existing Home active-work adapter seam.
- Kept the Schedules Console follow action disabled with explicit recovery copy when no active schedule run exists.
- Enabled `schedules-follow-in-console` when an active `Schedules` item has Console context.
- Routed Schedules Console follow through `open_active_home_item_in_console` so Home and destination surfaces share the same launch path.
- Updated Console source readiness so Schedules is shown as connected.

## Functional QA Evidence

Focused checks were run against the Schedules destination and Console handoff harness.

- Baseline before the slice: `python -m pytest -q Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_destination_shells.py --tb=short`
- Baseline result: `64 passed, 3 warnings`.
- Red Schedules Console launch test: `python -m pytest -q Tests/UI/test_console_live_work_handoffs.py --tb=short`
- Red result: `5 failed, 27 passed, 1 warning`; failures were expected because Schedules still rendered an unavailable Console action, source readiness still marked Schedules as `Not wired`, and this evidence/task tracking did not exist.
- Focused Console handoff verification after implementation: `python -m pytest -q Tests/UI/test_console_live_work_handoffs.py --tb=short`
- Focused Console handoff result: `32 passed, 1 warning`.
- Broader Phase 3 regression check: `python -m pytest -q Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_destination_shells.py Tests/UI/test_home_screen.py Tests/Home/test_active_work_adapter.py Tests/UI/test_subscription_window_watchlists.py Tests/UI/test_screen_navigation.py Tests/UI/test_unified_shell_phase2_home_adapter.py --tb=short`
- Broader Phase 3 result: `163 passed, 8 warnings in 76.80s`.

Warning boundary: warnings are existing dependency/import warnings and are not Schedules Console-launch behavior failures.

## UX Result

- First-time users still see an honest unavailable state when no schedule run can be followed.
- Power users can jump directly from Schedules to Console when an active schedule-run context exists.
- The Schedules surface now matches the W+C pattern instead of remaining a permanent false-negative disabled action.

## Residual Risk

- This slice wires the destination producer to adapter-provided context; it does not create a schedule-run backend snapshot adapter.
- Console still does not subscribe to real live event streams.
- Full Phase 3 still requires Workflows, ACP, MCP, RAG, and Artifacts live-work producers.
