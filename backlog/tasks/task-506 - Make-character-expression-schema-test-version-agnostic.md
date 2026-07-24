---
id: TASK-506
title: Make character expression schema test version-agnostic
status: Done
assignee: []
created_date: '2026-07-24 17:10'
updated_date: '2026-07-24 17:12'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the character-expression migration smoke test aligned with the database's declared current schema instead of a stale historical version literal.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Fresh database version equals CharactersRAGDB._CURRENT_SCHEMA_VERSION
- [x] #2 Character expression table behavior remains covered
- [x] #3 Focused character-expression tests pass
- [x] #4 No production schema or migration behavior changes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Record the stale hardcoded-version failure on branch and merge base. 2. Rename the test to describe the current-schema contract and assert the class-declared version. 3. Run the focused file, lint, and diff checks. 4. Record no-ADR rationale and complete the task only after verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Renamed the fresh-database schema test to describe its declared-current contract and replaced the stale literal 23 with CharactersRAGDB._CURRENT_SCHEMA_VERSION. Character-expression storage behavior remains unchanged and covered by the rest of the file. No production files were modified. ADR required: no; ADR path: N/A; Reason: test-only expectation hardening against an existing schema-version declaration, with no schema, migration, storage, or runtime decision. Root-cause evidence preserved: the broader fail-fast run stopped at this inherited assertion with 1 failed, 607 passed, and 1 skipped; the branch current version is 25, while the exact merge base current version is 24, so the v23 assertion already failed on base. Focused RED asserted 25 == 23; GREEN verification: full affected file 8 passed, Ruff check passed, Ruff format check passed, and git diff --check passed.
<!-- SECTION:NOTES:END -->
