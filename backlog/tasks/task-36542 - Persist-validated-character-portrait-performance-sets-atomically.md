---
id: TASK-36542
title: Persist validated character portrait performance sets atomically
status: Done
created_date: 2026-10-23 22:55
dependencies:
- TASK-36509
labels:
- character-chat
- database
documentation:
- Docs/superpowers/specs/2026-10-23-console-radio-portraits-design.md
- Docs/superpowers/plans/2026-10-23-console-radio-portraits-implementation.md
- backlog/decisions/242-console-radio-portrait-performance.md
- Docs/superpowers/reviews/2026-10-23-task-36542-review.md
- backlog/docs/lessons-portrait-performance.md
assignee:
- '@codex'
updated_date: 2026-10-24 00:17
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep a character's six-pose performance artwork durable and tied to the exact source expression, without modifying its original portrait or portable visual packs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Exactly six unique still-image roles normalize to PNG with matching dimensions; malformed, animated, oversized and over-budget inputs are rejected and decoded buffers are released.
- [x] #2 Fresh and upgraded character databases store one active set per character/expression; Apply and Remove are atomic, revision-checked and reject stale source or deleted owners.
- [x] #3 Revision identities remain distinct across remove and reapply, including identical source/frame bytes; stale clients cannot overwrite or mistake a replacement for its predecessor.
- [x] #4 Character soft delete/restore preserves attachments, permanent deletion removes their rows, and existing portraits/expression images remain byte-identical.
- [x] #5 Full-profile backup data includes the attachments while existing card, Actor Pack, Chatbook and sync formats remain unchanged.
- [x] #6 Targeted real-SQLite tests cover migration, rollback, stale concurrent writes, source mutation and purge; changed code passes applicable static checks.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/242-console-radio-portrait-performance.md
Reason: Implement ADR-242's local, atomic source-bound performance attachments; no new architectural decision.

1. Confirm live schema 81 and implement the executed V81→V82 portrait-performance migration, using historical_bootstrap for the preceding-version fixture.
2. Add and run failing normalization, source identity, migration and real-SQLite repository tests, including size limits, rollback, CAS races, revision tombstones and owner lifecycle.
3. Implement bounded six-still PNG normalization and canonical source capture, reusing installed artwork validation and database operation guards.
4. Add atomic Apply/get/Remove with source/revision revalidation, inactive revision preservation and permanent-delete cascades; retain original artwork, exports and sync behavior.
5. Run targeted new and existing expression/visual-identity/migration/backup checks, Ruff/format/import/packaging guards and scoped diff checks; update the character architecture doc.
6. Self-review the complete atomic change, correct findings, record evidence and implementation notes, then commit and mark Done before starting TASK-36543.
Scope traced before implementation: include DB/recovery_core_schema.py and DB/recovery_core.py's pinned schema lineage/migration policy, DB/sql_validation.py's new-table allowlist, Tests/Backup_Recovery/test_core_owners.py, and the current-schema expectations in Tests/DB/test_response_rules_migration.py. These are required callers of the V81→V82 migration, not new feature scope. Verify full-profile attachment inclusion through real SQLite backup/catalog admission now; full UI profile restore remains TASK-36550.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented ADR-242's profile-local portrait storage: six bounded aligned still PNG roles; shared preparation-budget/resource cleanup; canonical portrait/legacy/immutable source identity; atomic Apply/get/Remove with source and revision revalidation, revision tombstones and nested-write savepoints. Soft-delete/restore preserves rows, permanent deletion cascades, and original art/card/sync data remain unchanged.

Schema 81→82 is an executed packaged SQL migration. Updated DB/recovery_core_schema.py and recovery_core.py's fixed catalogs/policy, DB/sql_validation.py, current-schema tests and Docs/Architecture/character-chat.md. Added portrait_performance.py and three targeted normalization/repository/migration files. Recovery, allowlist and preceding-schema callers were necessary scope traced before implementation; no new export format or dependency.

Verification: initial missing-module red; nested caught-Apply red (revision 2 instead of 1), then green with Apply/Remove savepoints. Final new files plus response-rule migration callers: 40 passed. Broader new/existing expression, visual identity, migration and SQL guards: 179 passed. Native recovery policy/capture: 10 passed. Export regressions: 21 passed, 2 golden-byte failures reproduced identically on unchanged base export sources and fixtures; not introduced here. Importlib mode is required for Actor Pack relative imports. New files pass Ruff/format checks, zero introduced findings against exact per-file legacy HEAD baselines, 127-table guard passes, packaged migration glob covers V82, fresh import remains lazy, and scoped diff check passes.

Self-review completed before TASK-36543; no unresolved blocking finding. See Docs/superpowers/reviews/2026-10-23-task-36542-review.md. The nested-transaction incident is recorded in backlog/docs/lessons-portrait-performance.md. Full suite, installed full-profile UI restore and live rendering/audio/image quality remain later qualification obligations.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
