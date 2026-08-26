---
id: TASK-15707
title: Repair legacy ChaChaNotes migration test fixtures
status: Done
assignee: []
created_date: '2026-08-13 02:50'
updated_date: '2026-08-13 03:02'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the v20 and v21 ChaChaNotes migration tests construct genuine historical schemas so later schema additions do not invalidate the baseline migration suite.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The v20 and v21 tests seed migration-relevant historical world-book shapes without manually reversing unrelated current schema additions.
- [x] #2 The fixtures assert their historical column and trigger preconditions before reopening at the current version.
- [x] #3 The legacy migrations reach the current schema and preserve their world-book assertions.
- [x] #4 The focused migration baseline passes.
- [x] #5 ADR status is documented as not required because this is a test-only correction with no production architecture change.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add precondition assertions to the version-pinned v20/v21 seed fixtures and reproduce bootstrap-ahead world-book columns.
2. Normalize only `world_book_entries.priority`, `world_book_entries.regex`, and their two sync triggers to the corresponding v20/v21 historical shapes inside the scoped seed patches.
3. Reopen at the current version and verify the legacy migrations plus the broader Notes baseline.
4. Self-review the test-only correction and document ADR status and implementation notes.

ADR required: no
ADR path: N/A
Reason: test-fixture correction only; production schema, migration policy, and runtime behavior are unchanged.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Retained scoped `_CURRENT_SCHEMA_VERSION` pins, then normalized only the migration-relevant world-book bootstrap drift before closing each seed. The v20 fixture removes `priority` and `regex` and restores sync triggers that reference neither; the v21 fixture retains `priority`, removes `regex`, and restores priority-aware triggers that do not reference regex. Explicit column and trigger assertions verify these preconditions before reopening at the current version.

Root cause: the current bootstrap schema already contains the later world-book `priority` and `regex` columns. Pinning only the recorded version therefore skipped the v20→v21 and v21→v22 add-column branches, allowing recovery paths to make the tests pass without exercising the intended migrations. A RED run after adding the precondition assertions failed exactly because v20 retained `priority` and v21 retained `regex`.

Verification: RED: `.venv-test/bin/python -m pytest Tests/DB/test_chachanotes_world_book_priority_migration.py Tests/DB/test_chachanotes_world_book_regex_migration.py -q` → 2 failed, 2 passed (v20 `priority` / v21 `regex` still present). GREEN: the same focused command → 4 passed. Broader baseline: `.venv-test/bin/python -m pytest Tests/DB/test_chachanotes_console_context_memory_migration.py Tests/DB/test_chachanotes_world_book_regex_migration.py Tests/DB/test_chachanotes_world_book_priority_migration.py Tests/Notes/test_notes_scope_service.py Tests/Sync_Interop/test_notes_outbox_producer.py -q` → 51 passed.

Modified files: `Tests/DB/test_chachanotes_world_book_priority_migration.py`, `Tests/DB/test_chachanotes_world_book_regex_migration.py`, and this task record.

ADR required: no
ADR path: N/A
Reason: test-fixture correction only; production schema, migration policy, and runtime behavior are unchanged.
<!-- SECTION:NOTES:END -->
