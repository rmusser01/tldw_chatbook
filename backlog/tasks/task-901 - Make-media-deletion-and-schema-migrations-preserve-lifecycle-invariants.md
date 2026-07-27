---
id: TASK-901
title: Make media deletion and schema migrations preserve lifecycle invariants
status: Done
assignee:
  - '@codex'
created_date: '2026-07-24 06:34'
updated_date: '2026-07-24 07:13'
labels:
  - database
  - rag
  - migration
  - privacy
dependencies: []
references:
  - backlog/decisions/030-derived-index-lifecycle-and-atomic-media-migrations.md
documentation:
  - >-
    Docs/superpowers/plans/2026-07-24-media-lifecycle-and-migration-invariants.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ensure source-media deletion is reflected in derived semantic indexes and make media schema migrations fail atomically so retries cannot inherit partially applied schema.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Soft and hard media deletion remove the corresponding semantic-index document only after the source transaction commits
- [x] #2 Undeleting media re-enqueues the restored source for semantic indexing
- [x] #3 Missed deletion callbacks can be reconciled from durable indexing state without exposing deleted media
- [x] #4 Unavailable or failing optional RAG components never roll back or block source deletion
- [x] #5 A failed media schema migration leaves both schema objects and schema_version unchanged and can be retried successfully
- [x] #6 Real SQLite and vector-store regression tests verify deletion restore reconciliation and migration rollback behavior
- [x] #7 Focused tests static analysis and repository integrity checks pass
- [x] #8 A hard-delete batch failure rolls back all source and child-table mutations rather than committing a partial purge
- [x] #9 Moving media to trash removes its semantic projection and restoring it re-enqueues indexing
- [x] #10 RAG query caches are service-local and invalidated whenever indexing or removal changes searchable projections
- [x] #11 Media deletion and restoration preserve FTS and semantic-projection lifecycle when applied through remote sync
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing real SQLite and vector-store regressions for post-commit
   delete/restore lifecycle, missed-event reconciliation, failure containment,
   and mid-migration rollback.
2. Add post-commit media deletion events and queue derived-index removals; reuse
   post-ingest indexing for undelete.
3. Reconcile durable media indexing records against active source IDs during
   backfill.
4. Isolate RAG query-result caches per service and invalidate them whenever
   searchable projections change.
5. Execute every versioned media schema transition statement-by-statement
   inside the existing SQLite transaction boundary.
6. Preserve media FTS lifecycle across sync by emitting restorable update
   payloads and deriving receiver FTS and post-commit semantic lifecycle from
   the resulting source row.
7. Run focused and broader verification, self-review the diff, update task
   notes/checklists, and close the task.

ADR required: yes

ADR path:
`backlog/decisions/030-derived-index-lifecycle-and-atomic-media-migrations.md`

Reason: defines data ownership/lifecycle across source SQLite and derived
indexes and changes schema migration transaction contracts.

Detailed plan:
`Docs/superpowers/plans/2026-07-24-media-lifecycle-and-migration-invariants.md`
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the source/derived-index lifecycle boundary defined by
[ADR-030](../decisions/030-derived-index-lifecycle-and-atomic-media-migrations.md).
Media delete/trash now emits contained post-commit removal events, while
undelete/restore reuses post-ingest indexing. The indexer removes vector
documents before durable tracking rows, reconciles missed removals during
backfill, and invalidates a service-local query cache whenever searchable
projections change.

Hard-delete batches now fail atomically and notify projections only after
commit. Versioned media migrations execute complete SQLite statements inside
the caller-owned transaction so DDL, seed data, and `schema_version` roll back
together. The final review extended the original plan to repair the existing
sync boundary: restore events use the supported update operation with FTS
source fields, remote media changes rebuild/remove FTS transactionally, and the
resulting media state emits semantic lifecycle notifications after the sync
batch commits.

Regression coverage uses real SQLite plus the in-memory vector store and covers
callback ordering/failure containment, delete/restore/trash lifecycle,
reconciliation, cache isolation/invalidation, atomic migration retry, atomic
hard purge, and remote sync projection state. Final verification: 475 passed,
14 skipped across `Tests/Media_DB`, `Tests/RAG`, and `Tests/RAG_Admin`; Ruff,
Python compilation, and `git diff --check` passed. The one warning is the
environment's pre-existing Requests dependency-version warning.
<!-- SECTION:NOTES:END -->
