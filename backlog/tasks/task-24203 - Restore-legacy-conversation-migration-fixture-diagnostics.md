---
id: TASK-24203
title: Restore legacy conversation migration fixture diagnostics
status: Done
assignee:
  - '@codex'
created_date: '2026-08-29 13:36'
updated_date: '2026-08-29 13:38'
labels:
  - database
  - tests
  - migrations
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep isolated legacy conversation migration tests representative after database diagnostics switched from raw paths to stable fingerprints.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The legacy migration stand-in supplies the diagnostic state required by migration logging
- [x] #2 Legacy v12 and v13 migration parity tests pass without running unrelated schema initialization
- [x] #3 Targeted database regression tests and static checks pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A. Reason: this restores a test-only stand-in after an existing diagnostic privacy change; it does not change schema, migration policy, or runtime ownership. Confirm the fixture is the only migration stand-in missing diagnostic state, add the minimal privacy-preserving fingerprint field without invoking unrelated initialization, run the exact v13 failure plus the v12/v13 parity module and relevant migration tests, then Ruff/format/compile/diff checks before closing.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Updated the shared __new__-based legacy conversation migration fixture to initialize _db_diagnostic_ref with the same privacy-safe content fingerprint used by CharactersRAGDB.__init__. This keeps isolated migration tests independent of unrelated schema initialization while preserving the runtime logging contract. ADR required: no; ADR path: N/A. Verification: exact v13 plus both v12 legacy paths 3 passed; full chat/character parity modules 18 passed; diagnostic-path privacy module 36 passed; Ruff check passed; Ruff format check passed; compileall passed; git diff --check passed.
<!-- SECTION:NOTES:END -->
