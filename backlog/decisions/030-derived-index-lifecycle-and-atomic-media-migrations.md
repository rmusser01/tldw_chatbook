# ADR 030: Derived Index Lifecycle and Atomic Media Migrations

Status: Accepted
Date: 2026-07-24
Related Task: [backlog/tasks/task-901 - Make-media-deletion-and-schema-migrations-preserve-lifecycle-invariants.md](../tasks/task-901%20-%20Make-media-deletion-and-schema-migrations-preserve-lifecycle-invariants.md)
Supersedes: N/A

## Decision

The media SQLite database is authoritative for media lifecycle state. Semantic
indexes are derived projections: source writes commit first, then best-effort
post-commit lifecycle events enqueue index upserts or removals. Deleted and
trashed sources are inactive; undeleted and restored sources are active. Bulk
backfill must reconcile durable indexing records against active media so missed
removal events are repairable. Query-result caches are service-local and are
invalidated whenever projections change. Hard-delete batches are atomic. Every
versioned media schema transition, including its `schema_version` update and
seed data, executes in one real SQLite transaction; a failed transition rolls
back all of its DDL, data, and version changes.

## Context

Ingestion already emits a post-commit callback that asynchronously projects new
and updated media into the semantic vector store. Deletion, trash, and
restoration had no equivalent lifecycle contract. A real SQLite and in-memory
vector-store reproduction showed that setting `Media.deleted = 1` removed the
source from FTS while semantic search still returned the old `media_<id>`
document. Hard deletion also discarded the only source row without clearing
the projection.

The indexing-state database records successfully indexed source identifiers,
but backfill only visits active rows. Consequently, a missed or failed deletion
callback could persist indefinitely. This violates the privacy and retention
expectation that deleting or trashing a source makes it unavailable to
retrieval.

The same verification found that `undelete_media()` attempted to write the
unsupported sync operation `"undelete"` even though the schema permits only
create, update, delete, link, and unlink, so restoration always rolled back.
After changing restore to the supported update operation, review of the remote
consumer exposed a second contract break: media deletes left stale FTS rows,
media updates called an undefined helper, and the restore payload omitted the
title and content needed to reconstruct a missing FTS projection.
Hard purge attempted a second delete against an external-content FTS row that
soft deletion had already removed, producing an FTS5 corruption error. It also
swallowed per-item exceptions inside its outer transaction, allowing partial
purges to commit.

Media schema transitions wrap `sqlite3.Connection.executescript()` in the
database transaction context manager. Python's SQLite driver implicitly commits
any pending transaction before `executescript()`. An injected v2-to-v3 failure
therefore raised a migration error while leaving the first created table
persisted and `schema_version` unchanged. A retry can inherit a partial schema
and fail differently.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Delete vector chunks inside the source SQLite transaction | The vector store is a separate, optional persistence system with no shared transaction protocol. Its latency or failure must not block or roll back authoritative source deletion. |
| Treat deletion callbacks as sufficient without reconciliation | Process exit, optional dependency failures, and vector-store outages can lose best-effort events. Durable indexing state provides the repair set for eventual convergence. |
| Leave tracking rows after a successful vector deletion | Backfill would repeatedly retry already completed removals and could mistake deleted sources for current projections. |
| Bump `schema_version` before running migration DDL | A crash would advertise a schema that was never fully installed and prevent the normal migration chain from retrying. |
| Keep `executescript()` inside the generic transaction context | The driver-level implicit commit defeats the context manager's rollback boundary. Multi-statement migrations need transaction-preserving statement execution. |

## Consequences

Source deletion remains available when semantic indexing is disabled, missing,
or failing. Retrieval convergence is asynchronous: the committed source state
is immediate, while derived index cleanup completes through the worker or a
later reconciliation pass. Undelete and restore-from-trash use the same
post-commit indexing path as ingest, avoiding a second document-construction
contract. Their sync events use the existing update operation with explicit
`deleted = 0` or `is_trash = 0` payload state.

Lifecycle callbacks carry only stable media identity and action, execute after
commit, and contain observer exceptions. Successful projection removal deletes
the corresponding indexing-state row. Reconciliation compares tracked media
identifiers with active source identifiers and removes orphaned vector
documents before indexing active rows.

Media restore sync records use the supported update operation and include the
FTS source fields. The remote sync consumer derives media FTS state from the
authoritative row after each applied media mutation: active rows are upserted
and deleted rows are removed in the same SQLite transaction. After the sync
batch commits, the receiver emits the same semantic lifecycle notification
based on the resulting source row, including when conflict handling retained a
different but authoritative local state.

Hard purge relies on the preceding soft delete for FTS cleanup and performs no
invalid second FTS mutation. A failure anywhere in a purge batch rolls back all
source, child, and audit mutations; derived-index callbacks run only after the
batch commit.

Each RAG service owns its query-result cache. Sharing one process-global cache
across different vector stores or profiles can return another service's
documents and ignores later services' cache settings. Successful index upserts
and removals clear the owning service's query cache before durable indexing
state is finalized, while leaving the more expensive embedding cache intact.

Media migration scripts are executed statement-by-statement using SQLite's own
complete-statement parser while the normal transaction context is active. This
preserves trigger bodies and comments without introducing a new SQL parser, and
keeps migration DDL, seed rows, validation, and version updates inside one
rollback boundary.

## Links

- [Backlog task TASK-901](../tasks/task-901%20-%20Make-media-deletion-and-schema-migrations-preserve-lifecycle-invariants.md)
- [ADR-005: local RAG](005-invest-in-local-rag-mirroring-tldw-server.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-07-24-media-lifecycle-and-migration-invariants.md)
