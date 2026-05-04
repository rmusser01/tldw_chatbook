# Console Live-Work Source Readiness

Date: 2026-05-03
Task: `TASK-3.6`
Branch: `codex/unified-shell-phase3-next-slice`
Base: `origin/dev` at `551cd4d3`

## Purpose

Continue Phase 3 by making Console source readiness visible when no live-work item is staged. This avoids the UX gap where users can see a live-work card only after launch, but cannot tell which source families are currently connected.

## What Changed

- Added `ConsoleLiveWorkSourceReadinessState` and source readiness rows.
- Marked W+C as connected because Home W+C active work can now launch and route run details through Console.
- Marked Schedules as connected because Schedules can now follow active schedule-run context into Console when adapter context exists.
- Marked RAG as connected because Search/RAG results can stage selected evidence into Console.
- Marked Artifacts as connected because latest local Chatbook artifacts can launch into Console.
- Marked Workflows as connected because Workflows can launch active workflow-run context into Console when adapter context exists.
- Marked ACP and MCP as `Not wired` with explicit recovery copy.
- Updated `ChatScreen` to render the source readiness summary only when no pending Console live-work launch is staged.
- Preserved pending launch focus by suppressing the readiness summary while a live-work card is visible.

## Functional QA Evidence

Focused checks were run against the readiness model and mounted Textual Console screen harness.

- Red source-readiness test: `python -m pytest -q Tests/UI/test_console_live_work_handoffs.py --tb=short`
- Red result: `3 failed, 21 passed, 1 warning`; failures were expected because `ConsoleLiveWorkSourceReadinessState`, mounted source readiness rendering, and this evidence file did not exist.
- Green behavior test before evidence: `python -m pytest -q Tests/UI/test_console_live_work_handoffs.py --tb=short`
- Green behavior result: `1 failed, 23 passed, 1 warning`; the remaining failure was the intentionally missing evidence link.
- Focused Console verification after evidence: `python -m pytest -q Tests/UI/test_console_live_work_handoffs.py --tb=short`
- Focused Console result: `24 passed, 1 warning`.
- Broader Phase 3 regression check: `python -m pytest -q Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_home_screen.py Tests/Home/test_active_work_adapter.py Tests/UI/test_subscription_window_watchlists.py Tests/UI/test_screen_navigation.py Tests/UI/test_unified_shell_phase2_home_adapter.py Tests/UI/test_destination_shells.py --tb=short`
- Broader Phase 3 result: `155 passed, 8 warnings in 83.88s`.

Warning boundary: warnings are existing dependency/import warnings and are not Console source-readiness behavior failures.

## UX Result

- First-time users can see that W+C, Schedules, RAG, Artifacts, and Workflows are connected live-work sources.
- Power users can quickly distinguish supported Console follow-through from future source integrations.
- Planned sources remain visible without false affordances because unavailable rows are informational only.

## Residual Risk

- This is a source-readiness visibility slice, not full Phase 3 completion.
- ACP, MCP, and deeper source-specific live event producers still need implementation.
- Console still does not subscribe to real live event streams.
- Full Phase 3 cannot be verified until all relevant launch, follow, status, and recovery flows are exercised in the running app.
