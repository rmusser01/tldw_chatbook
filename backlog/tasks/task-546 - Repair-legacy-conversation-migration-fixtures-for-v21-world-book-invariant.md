---
id: TASK-546
title: Repair legacy conversation migration fixtures for v21 world-book invariant
status: Done
assignee:
  - '@codex'
created_date: '2026-07-24 20:22'
updated_date: '2026-07-26 07:45'
labels:
  - database
  - migrations
  - reliability
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make legacy v12/v13 conversation migration fixtures represent the historical schema so current migrations are tested without weakening fail-closed production behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Legacy v12 and v13 fixture databases include the world-book tables that existed from schema v9 without the v21 priority column
- [x] #2 Conversation parity migrations reach the current schema and preserve asserted rows
- [x] #3 The focused migration and full-suite fail-fast gates pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no

ADR path: N/A

Reason: This is a test-fixture correction that preserves the existing
migration boundary and ADRs.

1. Add a regression asserting the legacy fixture contains the historical
   pre-v21 world-book shape.
2. Extend the shared v12/v13 fixture with the v9 world-book tables and no
   priority column.
3. Run focused migration/parity tests, the diagnostic sentinel, and resume the
   full-suite fail-fast gate.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Summary: Restored the historical pre-v21 world-book schema in the shared legacy v12/v13 conversation fixtures so current migrations are exercised without weakening production fail-closed behavior.

Implementation:
- Added historical world_books, world_book_entries, and conversation_world_books tables to both fixture factories.
- Kept world_book_entries.priority absent so the v20-to-v21 migration must add it.
- Added a regression that verifies the required historical tables and columns and verifies priority is absent before migration.
- Existing conversation parity migrations continue to reach schema v21 and preserve their asserted rows.

Verification:
- Focused legacy fixture and conversation/runtime parity suite: 20 passed.
- Diagnostic/task sentinel harness: 2 passed.
- Final permitted full suite: 12,757 passed, 231 skipped, 240 warnings in 3h34m55s.
- Self-review: fixture-only change matches schema history; no production migration or fail-closed behavior changed.

ADR required: no
ADR path: N/A
Reason: Test-fixture correction preserving the existing migration boundary and ADRs.

Files modified:
- Tests/ChaChaNotesDB/legacy_conversation_schema.py
- Tests/ChaChaNotesDB/test_legacy_conversation_schema.py
- backlog/tasks/task-546 - Repair-legacy-conversation-migration-fixtures-for-v21-world-book-invariant.md
<!-- SECTION:NOTES:END -->
