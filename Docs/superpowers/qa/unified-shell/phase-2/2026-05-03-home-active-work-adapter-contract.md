# Home Active-Work Adapter Contract

Date: 2026-05-03
Task: `TASK-4.1`
Branch: `codex/unified-shell-phase2-home-adapter-contract`
Base: `origin/dev` at `499477a6`

## Purpose

Begin Phase 2 by replacing hard-coded Home active-work placeholder behavior with an explicit adapter boundary. This Home active-work adapter contract makes Home dashboard state and approve/reject/pause/resume/retry controls replaceable by real run, schedule, or agent services in later Phase 2 slices.

## What Changed

- Added `HomeActiveWorkAdapter`, `HomeControlAction`, `HomeControlResult`, and `HomeControlResultStatus`.
- Added `UnavailableHomeActiveWorkAdapter` as the honest default adapter.
- Wired `HomeScreen` dashboard input through `app.home_active_work_adapter`.
- Wired `TldwCli` approve, reject, pause, resume, and retry methods through the adapter.
- Preserved honest unavailable behavior: the default adapter notifies users that the action is not connected to an active run service yet and points them to details or Console.

## Running-App QA Evidence

Focused checks were run against pure Home state and mounted Textual Home tests.

- Red adapter test: `python -m pytest Tests/Home/test_active_work_adapter.py Tests/UI/test_home_screen.py::test_home_screen_uses_active_work_adapter_for_dashboard_and_controls -q`
- Red result before adapter existed: import error for `tldw_chatbook.Home.active_work_adapter`.
- Focused green check: `python -m pytest Tests/Home/test_active_work_adapter.py Tests/UI/test_home_screen.py::test_home_screen_uses_active_work_adapter_for_dashboard_and_controls -q`
- Focused green result after review follow-up: `4 passed`
- Home suite: `python -m pytest Tests/Home/test_active_work_adapter.py Tests/Home/test_dashboard_state.py Tests/UI/test_home_screen.py -q`
- Home suite result after review follow-up: `16 passed, 8 warnings`
- Evidence contract: `Tests/UI/test_unified_shell_phase2_home_adapter.py`

Warning boundary: warnings are existing dependency/import warnings and are not Home adapter behavior failures.

## UX Result

- Home now has a real adapter seam for status and control ownership.
- Beginner-facing copy remains honest when no active run service exists.
- Power-user paths are preserved because the visible controls and routes remain stable while the backend source becomes replaceable.

## Residual Risk

- This slice does not implement real approve/reject/pause/resume/retry services.
- Real active-run, schedule, and agent-service adapters remain future Phase 2 work under `TASK-4`.
- Home still needs running-app QA for real service-backed control completion before Phase 2 can be verified.
