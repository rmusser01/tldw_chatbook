# Home Detail And Console Adapter Actions

Date: 2026-05-03
Task: `TASK-4.2`
Branch: `codex/unified-shell-phase2-home-open-detail-adapter`
Base: `origin/dev` at `5a6e35e8`

## Purpose

Close the remaining false-control gap in the Phase 2 Home adapter seam. `Open details` and `Open in Console` now go through the active-work adapter rather than directly navigating from a visible button.

## What Changed

- Added `HomeControlAction.OPEN_DETAILS` and `HomeControlAction.OPEN_IN_CONSOLE`.
- Added `HomeConsoleLaunch` and optional `target_route` / `console_launch` fields to `HomeControlResult`.
- Wired Home `Open details` to navigate only after the adapter returns a handled detail target.
- Wired Home `Open in Console` to launch Console only after the adapter returns explicit Console launch context.
- Kept unavailable default behavior honest: missing active-work services notify with recovery copy and do not create a fake Console workflow.

## Functional QA Evidence

Focused checks were run against the Home adapter unit surface and mounted Textual Home screen harness.

- Baseline before edits: `python -m pytest Tests/Home/test_active_work_adapter.py Tests/UI/test_home_screen.py -q`
- Baseline result: `12 passed, 10 warnings`
- Red test: `python -m pytest Tests/Home/test_active_work_adapter.py Tests/UI/test_home_screen.py -q`
- Red result: collection failed because `HomeConsoleLaunch` and the open-detail/open-console adapter actions did not exist.
- Green test: `python -m pytest Tests/Home/test_active_work_adapter.py Tests/UI/test_home_screen.py -q`
- Green result: `18 passed, 8 warnings`
- Focused Phase 2 suite: `python -m pytest Tests/Home/test_active_work_adapter.py Tests/Home/test_dashboard_state.py Tests/UI/test_home_screen.py Tests/UI/test_unified_shell_phase2_home_adapter.py -q`
- Focused Phase 2 suite result: `26 passed, 8 warnings`

Warning boundary: warnings are existing dependency/import warnings and are not Home adapter behavior failures.

## UX Result

- First-time users no longer see a control that silently opens Console without a real active-work payload.
- Power users keep the same visible Home controls, but repeated detail/Console workflows now depend on explicit adapter results.
- Recovery remains understandable when live active-run services are not wired: the app tells users the action is unavailable instead of implying work is happening.

## Residual Risk

- This slice still uses the unavailable default adapter unless a real active-run/schedule adapter is provided.
- Service-backed detail records and Console launch payloads remain future Phase 2 work.
- Full Phase 2 cannot be verified until approve, reject, pause, resume, retry, detail, and Console workflows are exercised against real active work.
