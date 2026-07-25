---
id: TASK-638
title: Resolve private SQLite integration temp-root alias
status: Done
assignee: []
created_date: '2026-07-25 19:45'
updated_date: '2026-07-25 19:50'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the character-chat file-operation integration harness compatible with the private SQLite no-symlink invariant on platforms whose standard temporary directory is exposed through a lexical alias.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The database fixture passes a symlink-free trusted path to CharactersRAGDB
- [x] #2 The fixture closes its database connection after each test
- [x] #3 All character-chat file-operation integration cases pass
- [x] #4 Production path-validation and private-SQLite behavior remains unchanged
- [x] #5 Task notes record RED evidence, ADR decision, verification, and self-review
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the macOS setup failure and trace the rejected lexical path.
2. Resolve the standard-library temporary root before constructing the database path and make fixture teardown explicit.
3. Run the focused character-chat file-operation tests and the full integration file.
4. Run Ruff format/check and git diff --check, then self-review.

ADR required: no
ADR path: N/A
Reason: This corrects a test fixture to comply with the existing private-path contract and changes no production architecture or security policy.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Summary: Corrected the character-chat file-operation integration fixture to comply with the existing private SQLite no-symlink invariant and made database teardown explicit.

Approach and RED evidence:
- The focused integration run failed during fixture setup because tempfile exposed the macOS temporary root lexically under /var while private SQLite correctly rejected that symlinked ancestor with link_or_non_regular.
- Resolved the standard-library temporary root before constructing test.db, matching the established TASK-552 fixture pattern.
- Converted mock_db to a yielding fixture and close_connection() in finally, so every test closes the thread-local SQLite connection before the temporary directory is removed.
- Changed no production path-validation, SQLite, database, or character-chat code.

Verification:
- Both formerly failing integration files: 21 passed.
- Tests/integration Tests/test_reports Tests/tldw_api Tests/unit: 478 passed, 12 skipped.
- Ruff format check: 2 files already formatted.
- Ruff check: all checks passed.
- py_compile: passed.
- git diff --check: passed.
- Self-review: fixture dependency teardown closes mock_db before temp_base_dir cleanup; the change preserves the production privacy invariant and adds no bypass.

ADR required: no
ADR path: N/A
Reason: This is a test-fixture correction under the existing private-path contract, not a storage or security architecture change.

Files modified:
- Tests/integration/test_file_operations_with_validation.py
- backlog/tasks/task-638 - Resolve-private-SQLite-integration-temp-root-alias.md
<!-- SECTION:NOTES:END -->
