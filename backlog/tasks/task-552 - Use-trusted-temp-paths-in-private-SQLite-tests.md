---
id: TASK-552
title: Use trusted temp paths in private SQLite tests
status: Done
assignee: []
created_date: '2026-07-25 16:44'
updated_date: '2026-07-25 16:47'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep private SQLite-backed tests compatible with the no-symlink invariant on platforms where the standard temporary directory is exposed through an alias such as macOS /var.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Standard-library temporary-directory fixtures pass only symlink-free trusted paths to private SQLite
- [x] #2 Production private-path and SQLite code remains unchanged
- [x] #3 The focused Scheduling, smoke, and integration tests pass
- [x] #4 Task notes record the platform-specific RED evidence, ADR decision, and verification
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the isolated macOS RED failure.
2. Inventory the remaining private SQLite tests that construct database paths directly from the standard-library temporary directory.
3. Resolve those temporary roots before constructing database paths.
4. Run the focused Scheduling database file, full Scheduling suite, smoke tests, and affected integration test.
5. Review and document the invariant.

ADR required: no
ADR path: N/A
Reason: This is a test-fixture correction that preserves the existing private-path contract.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Summary: Resolved standard-library temporary roots before passing database paths into private SQLite-backed tests, preserving the lexical no-symlink policy on macOS and other aliasing platforms.

- Updated the Scheduling database fixture, database smoke tests, and the affected core integration test.
- Kept the unrelated path-validation smoke fixture unchanged because it does not open private SQLite.
- Changed no production private-path, SQLite, or database code.

RED evidence:
- `Tests/Scheduling/test_scheduled_tasks_db.py::test_get_schema_version` failed because macOS exposes the standard temporary directory under `/var`, a symlink to `/private/var`, and the private-path verifier correctly rejected the symlink during lexical descriptor traversal.
- `Tests/test_smoke.py::TestDatabaseSmoke::test_database_initialization` independently reproduced the same failure.

Verification:
- Focused Scheduling database, smoke, and core integration files: 51 passed, 1 skipped.
- Full `Tests/Scheduling`: 176 passed.
- Ruff check on all three changed test files: passed.
- `git diff --check`: passed.

ADR required: no
ADR path: N/A
Reason: This corrects test inputs to comply with the existing private-path contract and changes no production architecture or security policy.

Files modified:
- `Tests/Scheduling/test_scheduled_tasks_db.py`
- `Tests/test_smoke.py`
- `Tests/integration/test_core_functionality_integration.py`
- `backlog/tasks/task-552 - Use-trusted-temp-paths-in-private-SQLite-tests.md`
<!-- SECTION:NOTES:END -->
