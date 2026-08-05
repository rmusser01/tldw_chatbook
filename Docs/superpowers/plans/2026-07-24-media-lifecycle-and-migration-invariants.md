# Media Lifecycle and Migration Invariants Implementation Plan

Date: 2026-07-24
Task: [TASK-901](../../../backlog/tasks/task-901%20-%20Make-media-deletion-and-schema-migrations-preserve-lifecycle-invariants.md)
ADR: [ADR-030](../../../backlog/decisions/030-derived-index-lifecycle-and-atomic-media-migrations.md)

## Scope

Repair two verified invariants in the local media stack:

1. An inactive deleted or trashed media source must converge to absence from
   the derived semantic index without making the source transition depend on
   optional RAG availability.
2. A failed versioned media migration must leave no DDL, seed-data, or version
   residue and must remain retryable.

Provider configuration, eval execution, general worker contracts, packaging,
and application-state decomposition remain later tranches.

## Implementation Plan

1. Add real SQLite and vector-store regression tests that reproduce stale
   semantic retrieval after deletion/trash, post-commit callback ordering,
   undelete/restore re-indexing, missed-event reconciliation, observer failure
   containment, remote sync FTS delete/restore behavior, atomic hard purge, and
   partial migration DDL after an injected failure.
2. Extend the existing media post-commit callback seam with explicit lifecycle
   events for deletion/trash while reusing the post-ingest event for
   undelete/restore.
3. Add background deletion work to the ingestion indexer. Remove vector
   documents before deleting their durable indexing-state records, and contain
   failures so source transactions remain authoritative.
4. Reconcile tracked media identifiers against active source identifiers at the
   start of media backfill, removing orphaned projections before active rows are
   indexed.
5. Isolate query-result caches per RAG service and invalidate them after every
   successful projection upsert or removal.
6. Replace migration-time `executescript()` calls with a
   transaction-preserving statement executor based on
   `sqlite3.complete_statement()`, covering schema v0-to-v1 and every registered
   media migration.
7. Make media restore sync records use the supported update operation with the
   FTS source fields, and make the receiver derive FTS state from the resulting
   authoritative media row in the same transaction. Emit the matching semantic
   lifecycle event only after the sync batch commits.
8. Run focused database/RAG suites, static analysis, compilation, repository
   integrity checks, and broader regression tests proportional to affected
   boundaries. Review the resulting diff before closing TASK-901.

## ADR Check

ADR required: yes

ADR path:
`backlog/decisions/030-derived-index-lifecycle-and-atomic-media-migrations.md`

Reason: this tranche defines the authoritative-source/derived-index ownership
boundary and changes the transaction contract for versioned schema migrations.

## Verification Matrix

| Invariant | Evidence |
| --- | --- |
| Delete fires only after commit | Callback observes committed `deleted = 1`; rollback emits nothing |
| RAG failure cannot block source deletion | Failing callback/service test leaves source deleted |
| Soft and hard deletion remove projections | Real media DB plus vector-store tests |
| Undelete and untrash restore projections | Lifecycle worker round-trip test |
| Missed callbacks converge | Backfill reconciliation test over durable indexing state |
| Cached retrieval converges | Per-service isolation and mutation-invalidation tests |
| Remote sync preserves projection lifecycle | Delete removes FTS and semantic projections; restore recreates/re-enqueues both after commit |
| Migration failure is atomic | Injected mid-script failure leaves no probe table and old version |
| Migration retry works | Corrected script succeeds on the same database after the failure |
| Hard purge is atomic | Injected second-item failure preserves every source row |
