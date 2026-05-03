# Schedules Reading Digest Console Launch

Date: 2026-05-03
Task: `TASK-3.7`
Branch: `codex/unified-shell-phase3-schedules-console-launch`
Base: `origin/dev` at `f9783093`

## Purpose

Make the Schedules destination produce a real Console launch fallback when an actionable local reading-digest output exists and no active schedule-run context is available, while preserving an honest disabled state when no schedule context exists.

## What Changed

- `SchedulesScreen` now queries the existing local reading-digest output seam for the latest schedule output.
- `schedules-follow-in-console` stays disabled with recovery copy when no reading-digest output exists.
- When a reading-digest output exists, `schedules-follow-in-console` becomes enabled and identifies the output by title.
- Clicking the enabled Schedules Console launch action routes through `open_console_for_live_work`, preserving the existing Console live-work launch contract.

## Functional QA Evidence

Focused checks were run against Schedules destination fallback state, Schedules destination enabled state, click routing, and Phase 3.7 tracking evidence.

- Red test: `python -m pytest Tests/UI/test_console_live_work_handoffs.py::test_schedules_destination_keeps_console_launch_disabled_without_digest_output Tests/UI/test_console_live_work_handoffs.py::test_schedules_destination_routes_latest_digest_output_to_console Tests/UI/test_console_live_work_handoffs.py::test_phase3_schedules_console_tracking_evidence_links_task_and_roadmap -q`
- Red result: `3 failed`; failures covered no digest-output query, always-disabled Console launch, and missing Phase 3.7 evidence.
- Green behavior result: `2 passed, 1 warning in 5.75s`.
- Green targeted result after rebasing over PR #224: `3 passed, 1 warning in 5.59s`.
- Final focused Phase 3.7 suite after rebasing over PR #224: `141 passed, 8 warnings in 72.90s` for `Tests/UI/test_console_live_work_handoffs.py`, `Tests/UI/test_destination_shells.py`, `Tests/UI/test_home_screen.py`, `Tests/Home/test_active_work_adapter.py`, `Tests/UI/test_screen_navigation.py`, `Tests/UI/test_unified_shell_phase2_home_adapter.py`, and focused local reading-digest schedule service tests.

Warning boundary: warnings are existing dependency/import warnings and are not Schedules destination Console launch failures.

## UX Result

- Schedules no longer presents Console recovery as permanently unavailable when the app already has a local reading-digest output to inspect.
- No schedule output still produces a clear, non-clickable recovery state rather than a false affordance.
- The launch remains narrow: it stages the latest digest output in Console and does not pretend generic scheduler controls exist.

## Residual Risk

- This slice only covers local reading-digest outputs; generic workflow schedules and server-run schedule control remain deferred.
- Console primary follow-through for reading-digest output detail is not wired yet; the payload is staged for Console visibility and recovery.
- Pause, resume, retry, and delete controls for schedules remain outside this producer slice.
