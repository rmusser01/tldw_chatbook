---
id: TASK-642
title: Mark UI responsiveness async tests for root pytest
status: Done
assignee: []
created_date: '2026-07-26 03:43'
updated_date: '2026-07-26 03:44'
labels:
  - tests
  - pytest
  - asyncio
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the UI responsiveness async tests obey the repository-wide strict pytest-asyncio contract so they run in both the full-suite and nested UI configurations.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both async tests carry explicit pytest asyncio markers.
- [x] #2 The tests pass when forced through the repository-wide pyproject.toml pytest configuration.
- [x] #3 The full UI responsiveness module passes under both root and nested UI configurations.
- [x] #4 A test-tree audit finds no other collectable bare async test functions.
- [x] #5 Production code is unchanged.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the 85% full-suite failure and reproduce it with the repository-wide pytest configuration.
2. Audit the test tree for collectable async tests that rely on nested auto mode.
3. Add explicit asyncio markers to the two affected responsiveness tests.
4. Run root-config, nested-config, static, and full-suite verification.

ADR required: no
ADR path: N/A
Reason: This repairs test metadata for the existing repository-wide pytest contract and changes no application or architectural behavior.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Summary: Made both async UI responsiveness tests explicit participants in the repository-wide strict pytest-asyncio contract.

RED evidence and root cause:
- A fresh permitted full `pytest -q -x` run reached 10,908 passed and 216 skipped at 85%, then `test_console_sync_records_worker_lifecycle` failed because pytest did not run the bare coroutine.
- The repository-wide `pyproject.toml` does not enable pytest-asyncio auto mode, so the plugin uses strict mode. `Tests/UI/pytest.ini` enables auto mode only for UI-rooted invocations.
- Forcing the exact failing test through `-c pyproject.toml` reproduced the failure immediately, while the nested UI config passed it. No plugin-state mutation was involved.
- A syntax-tree audit of collectable module functions and `Test*` class methods found only the two bare async tests in this file; similarly named async methods on `Fake*` helper classes are not pytest tests.

Implementation:
- Imported pytest and added `@pytest.mark.asyncio` to both async tests.
- Sync tests in the module remain unmarked, avoiding inappropriate asyncio-marker warnings.
- No production code or pytest global configuration changed.

Verification:
- Full module under repository root config (`-c pyproject.toml`): 12 passed.
- Full module under nested UI auto config: 12 passed.
- Refined test-tree async-marker audit: no collectable bare async tests found.
- Ruff format check: already formatted.
- Ruff check: all checks passed.
- `py_compile`: passed.
- `git diff --check`: passed.
- Self-review: the smallest contract repair is local to the two coroutine tests and is valid in both strict and auto modes.

ADR required: no
ADR path: N/A
Reason: Test metadata repair only; no application or architectural behavior changes.

Files modified:
- Tests/UI/test_ui_responsiveness.py
- backlog/tasks/task-642 - Mark-UI-responsiveness-async-tests-for-root-pytest.md
<!-- SECTION:NOTES:END -->
