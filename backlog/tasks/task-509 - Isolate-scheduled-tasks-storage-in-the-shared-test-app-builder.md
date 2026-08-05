---
id: TASK-509
title: Isolate scheduled-tasks storage in the shared test app builder
status: Done
assignee: []
created_date: '2026-07-24 17:40'
updated_date: '2026-07-24 17:46'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent screen and Console tests using `_build_test_app` from touching cached or host scheduled-tasks database paths by routing that multi-connection database to the builder's isolated temporary directory.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `_build_test_app` routes scheduled tasks to an isolated temporary database.
- [x] #2 The agent runtime gate test passes alone and in its containing file.
- [x] #3 Screen navigation tests retain their isolated app behavior.
- [x] #4 No production storage configuration changes are made.
- [x] #5 Focused direct consumers of the shared test builder pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the read-only scheduled-tasks failure on branch and merge base.
2. Patch the shared `_build_test_app` scheduled-tasks path to its existing isolated temporary directory.
3. Run the failing Console case, screen navigation tests, and related shared-builder consumers.
4. Run lint, format, and diff checks; record the no-ADR test-isolation rationale and complete only after verification.

ADR required: no
ADR path: N/A
Reason: This is a test-isolation correction for an existing app-owned database and does not change production storage, schema, or runtime boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Isolated the scheduled-tasks database in the shared _build_test_app helper by patching get_scheduled_tasks_db_path to user_data_dir / scheduled_tasks.sqlite alongside its existing per-test database paths. The initial literal :memory: implementation made the exact regression pass but exposed SQLite connection isolation in mounted app tests: the scheduler opens a second connection and 8 screen-navigation cases failed with no such table: reminder_tasks. After approval, the isolated temporary file preserved the schema across connections while still preventing host or cached path access. No production files changed. RED: the exact test failed with attempt to write a readonly database. Verification after the final fix: exact regression 1 passed; full console-agent-swap plus screen-navigation files 88 passed; six additional direct-consumer files 29 passed; Ruff check passed; Ruff format check passed; git diff --check passed. Ruff also removed one pre-existing unused import and formatted three pre-existing long lines in the touched test module. ADR required: no. ADR path: N/A. Reason: test isolation only; production storage, schema, and runtime boundaries are unchanged.
<!-- SECTION:NOTES:END -->
