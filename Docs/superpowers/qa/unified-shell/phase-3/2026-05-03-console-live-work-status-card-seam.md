# Console Live-Work Status Card Seam

Date: 2026-05-03
Task: `TASK-3.2`
Branch: `codex/unified-shell-phase3-console-status-card`
Base: `origin/dev` at `09ca0200`

## Purpose

Continue Phase 3 by turning the first pending Console launch card into a reusable status-card display seam. The goal is to make future workflows, schedules, ACP, MCP, RAG, artifacts, and Home active-work sources render the same live-work status language without duplicating `ChatScreen` markup.

## What Changed

- Added `ConsoleLiveWorkStatusCardState` and `ConsoleLiveWorkStatusCardRow` as a stable display contract derived from `ConsoleLiveWorkLaunch`.
- Preserved the existing one-shot pending-launch behavior and existing user-visible copy.
- Added stable row selectors for source, title, status, recovery, action, and payload metadata.
- Moved `ChatScreen` pending launch rendering through a reusable `_render_console_live_work_status_card()` helper.

## Functional QA Evidence

Focused checks were run against the card-state model and a mounted Textual Console screen harness.

- Red card-state/rendering test: `python -m pytest -q Tests/UI/test_console_live_work_handoffs.py --tb=short`
- Red result: `2 failed, 12 passed, 1 warning`; failures were expected because `ConsoleLiveWorkStatusCardState` and stable row selectors did not exist.
- Green card-state/rendering test: `python -m pytest -q Tests/UI/test_console_live_work_handoffs.py --tb=short`
- Green result: `14 passed, 1 warning`
- Red tracking evidence test: `python -m pytest -q Tests/UI/test_console_live_work_handoffs.py --tb=short`
- Red tracking result: `1 failed, 14 passed, 1 warning`; failure was expected because this evidence file and roadmap links were not present.
- Final Console evidence test: `python -m pytest -q Tests/UI/test_console_live_work_handoffs.py --tb=short`
- Final Console evidence result: `15 passed, 1 warning`
- Final combined focused regression: `python -m pytest -q Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_home_screen.py Tests/Home/test_active_work_adapter.py Tests/Home/test_dashboard_state.py Tests/UI/test_unified_shell_phase2_home_adapter.py Tests/UI/test_screen_navigation.py Tests/UI/test_destination_shells.py --tb=short`
- Final combined focused result: `126 passed, 8 warnings`
- Whitespace check: `git diff --check`
- Whitespace check result: passed

Warning boundary: warnings are existing dependency/import warnings and are not Console status-card behavior failures.

## UX Result

- First-time users still see the same understandable pending Console launch card.
- Power users and future source integrations get a predictable display contract with stable selectors for fast regression coverage.
- This reduces the risk that workflows, schedules, ACP, MCP, RAG, or artifact sources each invent different status/recovery language.

## Residual Risk

- This is a rendering seam, not full Phase 3 completion.
- Console still does not subscribe to real live-work event streams.
- Source-specific launch/follow/status producers remain future Phase 3 work.
- Full Phase 3 cannot be verified until live or explicitly unavailable source flows are exercised in the running app.
