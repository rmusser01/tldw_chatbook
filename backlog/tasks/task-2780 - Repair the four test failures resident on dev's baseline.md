---
id: TASK-2780
title: Repair the four test failures resident on dev's baseline
status: Done
assignee: []
created_date: '2026-08-07 01:00'
labels:
  - tests
  - dev-baseline
  - uat
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The full-app UAT batch (PR #1378) catalogued four test failures that reproduce on pristine `origin/dev` — every future PR inherits them as red noise, which is exactly how the Lab ▸ Speech crash hid for weeks ("pre-existing noise" that was a real crash). Each was diagnosed to its introducing change; in all four cases the production behavior was a deliberate, merged change and the TEST encoded the stale expectation:

1. `test_console_sync_records_worker_lifecycle` — broke in the console-decomposition wave 2 merge (#1381): stages moved onto instance-held delegates (`_session`, `_workspace`) created in `__init__`, which `MagicMock(spec=ChatScreen)` cannot auto-stub (spec covers class attributes only). The task-1469 probe factory was built to survive delegation growth; this is delegation growth in a shape it couldn't see.
2. `test_conflicts_tab_renders_rows_and_resolves` — the Lab/Schedules/Logs UX overhaul (`9dd2374b5`, ADR-031) deliberately replaced per-button resolve tooltips with one guidance line; the test pinned the pre-overhaul copy.
3. `test_counts_returns_per_state_counts` — `IngestJobState.SKIPPED` joined the lifecycle (`40de4b136`, task-2220); `counts()` honors its documented every-state contract, the expected dict didn't.
4. `test_generate_image_handler_restores_draft_when_batch_raises` — `_clear_console_composer_draft` grew a `_sync_console_command_popup()` call after the `_bare_generation_screen` factory was written; on a detached screen that query dies with AttributeError past the QueryError guard.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- SECTION:ACCEPTANCE_CRITERIA:BEGIN -->
- [x] All four tests pass on the branch, and their containing files pass in full.
- [x] Each repair follows the introducing change's documented intent (recorded inline at the fix site), never reverting shipped behavior.
- [x] No production code changed.
<!-- SECTION:ACCEPTANCE_CRITERIA:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Test-only changes, one per failure: (1) the sync-probe factory stubs the two instance-held delegates with a comment explaining the spec-mock blind spot; (2) the conflicts test pins the shipped ADR-031 guidance copy; (3) the counts expectation gains `"skipped": 0` citing task-2220; (4) the bare-screen factory stubs `_sync_console_command_popup` alongside its existing sync stub. Full files green: 15 + 39 + 66 + 35. Files: Tests/UI/test_ui_responsiveness.py, Tests/UI/test_schedules_workbench.py, Tests/Library/test_library_ingest_jobs.py, Tests/Chat/test_console_generation_actions.py.
<!-- SECTION:NOTES:END -->
