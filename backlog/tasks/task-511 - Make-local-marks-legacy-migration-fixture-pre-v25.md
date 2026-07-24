---
id: TASK-511
title: Make local-marks legacy migration fixture pre-v25
status: Done
assignee: []
created_date: '2026-07-24 18:09'
updated_date: '2026-07-24 18:12'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the local-marks v16 migration test's historical fixture contract after citation provenance became schema v25, without weakening the production migration's fail-closed handling of pre-existing provenance tables.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The local-marks migration fixture removes schema objects that did not exist at v16 before rolling its recorded version back
- [x] #2 The v16-to-current replay recreates local marks and citation provenance successfully
- [x] #3 Production citation migration continues to reject unexpected pre-existing or partial provenance tables
- [x] #4 The affected test file and citation migration atomicity coverage pass
- [x] #5 Task documentation includes the ADR decision, verification evidence, and implementation notes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the isolated failure on the feature branch and the passing merge-base result.
2. Correct only the legacy test fixture so it removes v25 provenance objects before recording v16.
3. Run the local-marks test file plus citation migration fail-closed/atomicity coverage.
4. Run Ruff format/check and git diff --check; independently review before completion.

ADR required: no
ADR path: backlog/decisions/024-rag-citation-provenance-and-source-resolution.md
Reason: ADR-024 already defines the citation schema and fail-closed migration contract; this task only repairs a historical test fixture to represent a true pre-v25 database.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a test-fixture-only correction for the local-marks v16 migration replay. The fixture now removes all 12 citation-provenance v25 tables in their established foreign-key-safe order before recording schema version 16; reopening the database therefore exercises the real v16-to-current migration path. Production migration behavior and schema code were not changed.

ADR decision: no new ADR required. Existing ADR-024 (`backlog/decisions/024-rag-citation-provenance-and-source-resolution.md`) defines the v25 schema and fail-closed/atomic migration contract; this change restores the fixture to that contract.

Verification evidence:
- Branch RED before the fix: exact local-marks migration test failed with `table rag_identity_context already exists`; the handoff also supplied a passing merge-base result for the same test.
- Exact repaired test: 1 passed.
- `Tests/Chat/test_conversation_local_marks_service.py`: 14 passed.
- `Tests/DB/test_chachanotes_citation_provenance_migration.py`: 14 passed, including fail-closed and rollback coverage.
- Ruff check passed; Ruff format check reported the owned test already formatted; `git diff --check` passed.
<!-- SECTION:NOTES:END -->
