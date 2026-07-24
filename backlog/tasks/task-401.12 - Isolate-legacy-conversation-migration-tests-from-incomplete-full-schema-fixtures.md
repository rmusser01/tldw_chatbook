---
id: TASK-401.12
title: >-
  Isolate legacy conversation migration tests from incomplete full-schema
  fixtures
status: Done
assignee: []
created_date: '2026-07-24 16:42'
updated_date: '2026-07-24 16:48'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make legacy conversation parity tests exercise the specific v12-to-v14 conversation migrations without presenting a conversations-only partial database to the full schema migrator.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Legacy v12 conversation rows verify scope and assistant-identity backfills through the v12-to-v13 migration
- [x] #2 Legacy v13 rows verify runtime and discovery defaults through the v13-to-v14 migration
- [x] #3 Tests no longer rely on a partial database reaching current schema
- [x] #4 Production migration validation remains unchanged
- [x] #5 ChaChaNotesDB and DB verification suites pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the three failing base cases as RED evidence. 2. Add a minimal test harness that invokes only the conversation migration under test. 3. Replace current-schema repository reads with direct migrated-row assertions. 4. Run focused tests, the full ChaChaNotesDB plus DB slice, lint, and diff checks. 5. Record no-ADR rationale and implementation notes before completion.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replaced full CharactersRAGDB initialization for the handcrafted v12/v13 conversation fixtures with a context-managed test harness built around CharactersRAGDB.__new__. The harness opens a real sqlite3.Row connection and invokes only the selected production v12-to-v13 or v13-to-v14 migration. The three tests now read migrated conversation rows directly, preserving exact scope, assistant-identity, runtime/discovery backfill expectations and the no-conversation-sync-log assertion. No production code, migration validation, or unrelated schema was changed. ADR required: no; ADR path: N/A; Reason: test-only isolation of existing migration contracts with no architecture, schema, storage, or runtime boundary decision. TDD evidence: the three focused tests failed before the correction when full initialization reached v20 and rejected missing world_book_entries, then passed through the direct harness. Verification: focused regressions 3 passed; both affected files 18 passed; full Tests/ChaChaNotesDB/ plus Tests/DB/ slice 332 passed; Ruff lint and format checks passed; git diff --check passed. TASK-401.11 and TASK-401.12 are intentionally standalone because the repository contains duplicate TASK-401 IDs and an explicit parent reference resolves ambiguously; both tasks now have separate acceptance-criterion checkboxes.
<!-- SECTION:NOTES:END -->
