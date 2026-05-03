# Console Live-Work Launch Contract

Date: 2026-05-03
Task: `TASK-3.1`
Branch: `codex/master-shell-ux-next-slice`
Base: `origin/dev` at `52e4e89e`

## Purpose

Start Phase 3 by replacing ad hoc pending Console launch dictionaries with a typed launch contract. This gives Home, workflows, schedules, ACP, MCP, RAG, artifacts, and future live-work sources one consistent payload shape before deeper Console live-status rendering is added.

## What Changed

- Added `ConsoleLiveWorkLaunch` as the normalized app-owned launch contract.
- Kept `TldwCli.open_console_for_live_work(source=..., title=..., payload=...)` backward compatible while adding optional `status`, `recovery`, and `action_label` metadata.
- Updated `ChatScreen` to consume typed launches or legacy dicts, clear the one-shot pending app value, and render source, title, status, recovery copy, action label, and top-level payload metadata.
- Preserved staged-context handoffs as separate `ChatHandoffPayload` flows so Library, Artifacts, Personas, and Skills do not masquerade as live work.

## Functional QA Evidence

Focused checks were run against the app helper and a mounted Textual Console screen harness.

- Red test: `python -m pytest -q Tests/UI/test_console_live_work_handoffs.py --tb=short`
- Red result: `4 failed, 9 passed, 1 warning`; failures were expected because `tldw_chatbook.Chat.console_live_work` did not exist.
- Green test: `python -m pytest -q Tests/UI/test_console_live_work_handoffs.py --tb=short`
- Green result: `13 passed, 1 warning`
- Expanded Console/Home/navigation regression: `python -m pytest -q Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_home_screen.py Tests/Home/test_active_work_adapter.py Tests/Home/test_dashboard_state.py Tests/UI/test_unified_shell_phase2_home_adapter.py Tests/UI/test_screen_navigation.py --tb=short`
- Expanded regression result: `82 passed, 8 warnings`
- Destination shell regression: `python -m pytest -q Tests/UI/test_destination_shells.py --tb=short`
- Destination shell result: `35 passed, 1 warning`
- Final combined focused regression: `python -m pytest -q Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_home_screen.py Tests/Home/test_active_work_adapter.py Tests/Home/test_dashboard_state.py Tests/UI/test_unified_shell_phase2_home_adapter.py Tests/UI/test_screen_navigation.py Tests/UI/test_destination_shells.py --tb=short`
- Final combined result: `117 passed, 8 warnings`
- Whitespace check: `git diff --check`
- Whitespace check result: passed

Warning boundary: warnings are existing dependency/import warnings and are not Console launch behavior failures.

## UX Result

- First-time users see an explicit pending Console launch card instead of unexplained context transfer.
- Power users keep the fast one-call launch helper, with richer metadata available for repeated live-work flows.
- Recovery language is visible at the point of handoff, which avoids implying that a workflow is actively running when the source only staged a request.

## Residual Risk

- This is a contract slice, not full Phase 3 completion.
- Source-specific live work from workflows, schedules, ACP, MCP, RAG, and artifacts still needs service-backed payload producers.
- Console still needs richer live status/follow cards after real source events exist.
- Full Phase 3 cannot be verified until launch, follow, status, and recovery flows are exercised in the running app against real or explicitly unavailable source adapters.
