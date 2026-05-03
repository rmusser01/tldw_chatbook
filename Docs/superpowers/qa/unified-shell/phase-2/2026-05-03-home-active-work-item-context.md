# Home Active-Work Item Context

Date: 2026-05-03
Task: `TASK-4.3`
Branch: `codex/unified-shell-phase2-home-dashboard-snapshot`
Base: `origin/dev` at `726f4954`

## Purpose

Make Home active-work controls less ambiguous by giving the dashboard an explicit visible work-item model. Home can now show what item is being acted on and pass that item's `target_id` through the app hooks into the active-work adapter.

## What Changed

- Added `HomeActiveWorkItem` with `item_id`, `title`, `source`, `status`, `detail_route`, and `console_available`.
- Added `target_id` to `HomeControl` and `HomeControlResult`.
- Derived Home controls from active-work item status when items are present.
- Preserved count-only fallback behavior for unavailable and legacy adapter states.
- Passed `target_id` through HomeScreen, app-level Home control hooks, and `HomeActiveWorkAdapter.handle_control`.

## Functional QA Evidence

Focused checks were run against pure Home dashboard state, the adapter contract, and the mounted Textual Home screen harness.

- Baseline before edits: `python -m pytest Tests/Home/test_dashboard_state.py Tests/Home/test_active_work_adapter.py Tests/UI/test_home_screen.py Tests/UI/test_unified_shell_phase2_home_adapter.py -q`
- Baseline result: `26 passed, 10 warnings`
- Red test: `python -m pytest Tests/Home/test_dashboard_state.py Tests/Home/test_active_work_adapter.py Tests/UI/test_home_screen.py -q`
- Red result: collection failed because `HomeActiveWorkItem` did not exist.
- Green test: `python -m pytest Tests/Home/test_dashboard_state.py Tests/Home/test_active_work_adapter.py Tests/UI/test_home_screen.py -q`
- Green result: `24 passed, 8 warnings`
- Status-gating regression: `python -m pytest Tests/Home/test_dashboard_state.py::test_dashboard_item_statuses_gate_matching_controls -q`
- Status-gating red result: failed because an approval-only item incorrectly exposed `home-pause`.
- Focused Phase 2 suite: `python -m pytest Tests/Home/test_dashboard_state.py Tests/Home/test_active_work_adapter.py Tests/UI/test_home_screen.py Tests/UI/test_unified_shell_phase2_home_adapter.py -q`
- Focused Phase 2 suite result: `31 passed, 8 warnings`

Warning boundary: warnings are existing dependency/import warnings and are not Home item-context behavior failures.

## UX Result

- Home can show visible active-work context instead of only aggregate counts.
- Approve, reject, pause, resume, retry, details, and Console actions can now target a specific active-work item.
- Explicit item context reduces accidental approval or recovery actions when multiple services eventually feed Home.

## Residual Risk

- This slice does not yet provide a real service-backed active-work feed.
- Real approval, schedule, workflow, and agent adapters still need to populate `HomeActiveWorkItem` records.
- Full Phase 2 verification still requires running-app QA against real service-backed work items.
