---
id: TASK-41534
title: >-
  Settings refused-custom-endpoint UI test polls the status line mid-recompose
  and fails with NoMatches
status: Done
assignee:
  - '@claude'
created_date: '2026-11-17 06:57'
labels:
  - settings
  - tests
  - ci
  - flaky
priority: high
references:
  - Tests/UI/test_settings_custom_endpoint_default.py
  - tldw_chatbook/UI/Screens/settings_screen.py
  - tldw_chatbook/UI/Settings_Modules/providers_models_card.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/UI/test_settings_custom_endpoint_default.py::test_settings_lists_a_refused_endpoint_with_its_reason_and_edit_fixes_it` (TASK-40300) fails with `NoMatches: No nodes match '#settings-custom-endpoints-status'`. It failed twice in UI Fast Lane (15) on run 44017381806 and on dev locally, and the merge queue evicts every PR after two such failures. The status line's id has not changed since TASK-33007.1, and the line is not lost: the test reads it at a moment when it is briefly absent. Fix the test's wait, without weakening what it proves: a refused endpoint is listed with its reason, and Edit fixes it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The cause is named, with the commit that introduced it, and shown to be test drift or a product defect with evidence
- [x] #2 The node passes 3 times in a row alone and 3 times in its whole-file run, at the load the shared machine has
- [x] #3 The test still asserts that the refused endpoint is listed with the contract's reason (never the value), has no Duplicate, and that Rename and then Edit save and the fixed entry loads
- [x] #4 Reverting only the fix turns the node red again
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (a test-only wait fix; no product behaviour changes).

1. Reproduce the failure alone and in the file; read the traceback and the child log.
2. Check whether the id or the Settings flow changed since TASK-40300 (`git log -S`, diff of the custom-endpoint path), and whether the status line is gone for good or only for a moment.
3. Fix where every caller routes through (`_click_until`), keeping the final check strict.
4. Prove it: node x3, file x3, mutation check, base-vs-head covering files, preflight, UI census.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Cause: test drift, not a product defect.** 06c2a0b374 (TASK-40300 review r14ep-rev6) replaced "wait for workers, settle, read the status once" with `_click_until`, which polls `done()` every 50 ms. Both save predicates read `#settings-custom-endpoints-status` through `query_one`. Every save reports through `_custom_endpoints_report`, which recomposes the whole `#settings-custom-endpoints` region, status line included. Textual's `Widget.recompose` awaits the removal of the old children before it mounts the new ones (inside `batch()`, so nothing is painted in between). A poll that lands in that gap raises `NoMatches`, and `_click_until` did not catch it. The id has not changed since TASK-33007.1 (9bad31b7f9), and no code on this path changed after 350cdc4c77. The gap is wider under load, which is why the r14ep-rev6 runs at load 12-35 passed and dev at load 55-100 does not.

**Evidence.**
- Unmodified dev 5e3c8b2bd4, node alone: 2 of 3 failed, at line 752 (edit save) and line 741 (rename save). CI run 44017381806 shows the same trace.
- A scratch probe (not committed) called `_custom_endpoints_report` three times and read the status on every loop turn. Each time the line was absent for 5 consecutive turns, then came back with the new message. Same result in 2 of 2 runs, so the row is never lost.

**Fix.** In `_click_until`, a `QueryError` from `done()` while polling counts as "not yet". The final `assert done()` still queries the status for real, so a status line that never comes back still fails the test. No assertion changed.

**Proof (load 55-105).**
- Node alone: 3/3 passed.
- Whole file, serial as in the UI Fast Lane: 57 passed, three times.
- Mutation (fix reverted only): 3 of 4 failed, all `NoMatches` on the status line. Fix re-applied.
- Other covering files at `-n 4` (`test_settings_custom_endpoints.py`, `test_settings_endpoint_probe.py`, `test_console_endpoint_discovery.py`, `Tests/Chat/test_custom_endpoint_{engine_swap,registry,trace_surface}.py`):
  - Base: 10 failed, 228 passed. Head: 11 failed, 227 passed.
  - The same 10 fail on both sides: 9 local `RecoveryRequired: raw_source_selection_changed` and 1 timing case.
  - The head-only `test_create_endpoint_with_live_controller_rebase_settles` passes alone on head and on base (2/2 each).
- `./scripts/preflight.sh` passed. `check_ui_pr_gate_census.py`: OK (2147). `check_backlog_task_ids.py`: OK.

**Files:** `Tests/UI/test_settings_custom_endpoint_default.py`; lesson added to `backlog/docs/lessons-textual.md`.
<!-- SECTION:NOTES:END -->
