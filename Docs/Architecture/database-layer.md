# Database layer

This document describes the SQLite layer: the `BaseDB` infrastructure (private connections, quiescence, semantic-mutation guards), the ChaChaNotes schema and its file-backed migrations, the Media DB and its in-code migrations, the cross-DB patterns (transactions, optimistic locking, soft deletion, FTS5), and the satellite databases.

## Authoritative files

| File | Role |
| --- | --- |
| `DB/base_db.py` | `BaseDB` ABC (path handling, `':memory:'`, client ids, private connections, vacuum, integrity check) plus process-wide machinery: `SQLiteConnectionQuiescenceRegistry`, quiescent connection/cursor wrappers, `_SemanticMutationAuthorization` + `register_semantic_mutation_guard` |
| `DB/ChaChaNotes_DB.py` | `CharactersRAGDB` — conversations, messages, characters, notes, keywords, world books, learning tables, console policy columns. Schema version is defined by `_CURRENT_SCHEMA_VERSION` |
| `DB/Client_Media_DB_v2.py` | `MediaDatabase` — media items, chunks, transcripts, document versions, chunking templates. Schema v9 |
| `DB/migrations/chachanotes_vN_to_vN+1_*.sql` | File-backed ChaChaNotes migration steps (v16→v67+ present) |
| `DB/store_kernel.py` | `SQLiteStoreKernel` mixin — the one held-connection/transaction implementation (ADR-228): autocommit held thread-local connections, 30s liveness probe, BEGIN IMMEDIATE `transaction()` with BaseException rollback, participants wiring. `RAG_Indexing_DB` and `Workspace_DB` consume it; remaining stores migrate one per PR |
| `DB/sql_validation.py` | Identifier validation (`escape_identifier`, `validate_table_name`, `validate_column_name`) |
| `DB/Prompts_DB.py`, `Evals_DB.py`, `Subscriptions_DB.py`, `AgentRuns_DB.py`, `Library_*_DB.py`, `RAG_Indexing_DB.py`, `Workspace_DB.py`, `VisualIdentity_DB.py`, `Chunking_Lab_DB.py` | Satellite databases |

Note: the "schema v37" figure that circulates in older docs is stale — read `_CURRENT_SCHEMA_VERSION` in `DB/ChaChaNotes_DB.py` for the current version.

## BaseDB infrastructure

- **Private connections** (`connect_private_sqlite`) harden the SQLite build for user-data files.
- **Quiescence registry**: an exclusive maintenance barrier per canonical file identity. Cursors hold use reservations until result exhaustion; `quiesce_connections()` drains and closes same-file handles so maintenance (vacuum, migration) can run without "database is locked" races.
- **Semantic-mutation guards**: `register_semantic_mutation_guard` registers SQLite functions plus a trace callback and an authorizer that **denies COMMIT/ROLLBACK escape mid-scope** — fail-closed guards consumed by DB triggers protecting console trace/semantic state.

## ChaChaNotes (CharactersRAGDB)

### Schema

Core tables: `character_cards`, `conversations` (TEXT UUID PK, `root_id`, `forked_from_message_id`, character FK), `messages` (with `parent_message_id` for the branch tree), `keywords` / `keyword_collections` + link tables, `notes` (with folders and organization sync ids), `research_quick_note_owner_proofs` (local-only), `sync_log` (trigger-populated change feed for server sync), plus later additions: `world_books`, learning/flashcard tables, `message_attachments`, `message_generation_metadata`, canvas revisions, and the console policy/memory columns (`console_project_context_json`, `console_conversation_memories`, `active_leaf_message_id`).

### Migrations

Steps are `.sql` files under `DB/migrations/`, executed **statement-by-statement through a cursor inside the step transaction** (an earlier `executescript` approach auto-committed and was replaced). Idempotent-replay guards handle re-runs: require-entry-version checks, skip-already-applied `ADD COLUMN`, drop-superseded-trigger. The runner is kept aligned with the folder by test.

### Patterns (the load-bearing conventions)

- **Transactions**: `with db.transaction(*, immediate=False) as cur:` — thread-local depth counter; the outermost context issues `BEGIN`/`BEGIN IMMEDIATE`. `immediate=True` avoids SQLite's non-retryable deferred-upgrade deadlock for read-then-write flows. Nested contexts defer to the outer; a caller-owned native transaction is borrowed without committing it.
- **Optimistic locking**: mutable rows carry `version`; updates/deletes guard `WHERE id = ? AND version = ?`; mismatches raise `ConflictError`; unique-constraint `IntegrityError` maps to `ConflictError` too.
- **Soft deletion**: `deleted` columns; FTS triggers index only live rows.
- **FTS5**: external-content virtual tables (`notes_fts`, `messages_fts`, `character_cards_fts`) maintained by `_ai/_au/_ad` triggers; query building goes through `Utils/fts5_match_forms` (prefix/AND forms, searchable checks). ChaChaNotes FTS is **trigger-driven** — contrast the Media DB below.
- **Parameterized everything**: identifiers only through `sql_validation`; values always bound; debug param previews are lazy and redactable.

## Media DB (Client_Media_DB_v2)

Tables: `Media` (unique url/content_hash/uuid, `vector_embedding` BLOB, `chunking_status`, version lineages with `prev_version`/`merge_parent_uuid`), `Keywords`/`MediaKeywords`, `Transcripts` (unique per `(media_id, whisper_model)`), `MediaChunks`, `UnvectorizedMediaChunks` (with `chunk_engine_version`, `chunking_template` stamps), `DocumentVersions`, `sync_log`, `ReadingProgress`, read-it-later state, and `ChunkingTemplates` (with server built-ins). Migrations are an **in-code registry**, not SQL files. FTS (`media_fts`, `keyword_fts`) is maintained **in Python**, not triggers.

The single write seam is `add_media_with_keywords`: URL/content-hash dedup with an `overwrite` flag, opt-in `restore_trashed` (only the Library ingest writer passes it — other duplicate matches against trashed rows are skipped), and post-commit post-ingest callbacks that feed RAG indexing.

## Satellite databases (one-liners)

- `Prompts_DB` — prompts + keywords, own migrations to v4, Python-side FTS, soft deletes, expected-version conflicts, Prompt/Recipe discriminator.
- `Evals_DB` — eval tasks/models/runs/results + FTS; version tracked via `PRAGMA user_version` (no version table).
- `Subscriptions_DB` — watchlist/subscription state + briefing provenance; FTS-with-LIKE-fallback at the search boundary ("the search box must never raise into the reader").
- `AgentRuns_DB` — agent run/step records (run trees; outliving sub-agent rows).
- `Library_Collections_DB` / `Library_Ingest_Jobs_DB` — collections and the durable ingest job queue.
- `RAG_Indexing_DB` — incremental indexing state (see [rag.md](./rag.md)).
- `Workspace_DB`, `VisualIdentity_DB`, `Chunking_Lab_DB` — workspace bindings, visual identity, chunking experiments.

Database helpers in `config.py` resolve explicit path overrides first, then profile-aware defaults under `get_user_data_dir()`. Eval private configuration follows the separate effective-config-owned selector described in [evals.md](./evals.md).

## Failure behaviors

| Case | Behavior |
| --- | --- |
| Version mismatch on update/delete | `ConflictError` (optimistic locking) |
| Corrupt/wrong-version store (permission store JSON, not SQLite) | `.bak` + fresh default — never raise out of load |
| FTS5 unavailable | LIKE fallback at search boundaries (Subscriptions pattern) |
| Migration crash mid-step | Step transaction rolls back; statement-by-statement keeps earlier statements of the same step atomic with it |
| Seeded character row vs FTS triggers | The seed row must be inserted after its triggers exist or the first edit raises `SQLITE_CORRUPT_VTAB` (companion repair exists for old DBs) |

## Verified gotchas

1. ChaChaNotes migrations are `.sql` files; the Media DB's are an in-code registry — two different mechanisms in one app.
2. ChaChaNotes FTS is trigger-driven; Media DB FTS is Python-driven.
3. `execute_query(commit=True)` only commits when the connection is not already inside a managed transaction.
4. Only the Library ingest writer restores trashed duplicates; every other writer skips them.
5. The branch tree (`parent_message_id`) plus the local-only `active_leaf_message_id` pointer is what swipes and rewind navigate — see [chat-pipeline.md](./chat-pipeline.md).

## Related docs

- [chat-pipeline.md](./chat-pipeline.md) — the messages/conversations consumers
- [notes-sync.md](./notes-sync.md) — the notes tables' sync authority
- [media-ingestion.md](./media-ingestion.md) — the Media DB write seam

## AgentRuns execution and inspection (ADR-244)

File-backed `AgentRunsDB` construction now requires a current
`ConsoleExecutionOwner` before schema initialization or orphan reconciliation.
All mutation methods, transaction entry points (including automatic-work/Goals),
and vacuum require the same owner. In-memory isolated stores require no OS lease.
`AgentRunsDB.open_read_only(path)` opens an existing store through the
`db.agent_runs.read` private-SQLite policy and participants provenance; it never
initializes/migrates schema or sweeps runs, and refuses mutation/recovery.

The stable adjacent lock is never unlinked. After winning it, native private-path
admission securely hardens a trusted owned legacy file to 0600; symlinks, multiple
links, foreign ownership and unsafe parents fail closed. A contender cannot
harden, create or sweep the run store. Ownership is independent of maintenance
admission, which continues to gate all native connections.


## Console remote command evidence (ADR-244)

AgentRuns v25 executes `agent_runs_v24_to_v25_console_command_receipts.sql`
through the native owner and the exact backup migration registry. Its
`command_receipts` repository uses the existing execution lease and SQLite
transaction owner; read-only handles can query without migrating or sweeping.
Rows contain identities, keyed payload fingerprints, monotonic issuance/expiry,
native outcome state and fixed reason codes, never submitted text.

An atomic `claim` returns explicit `inserted` ownership. A pending row does not
permit dispatch. The internal total bound is 4,096 receipts: capacity refuses
new non-idempotent effects. Pre-claim pruning can remove known acceptance,
queue, application or refusal evidence only after expiry in that same owner
clock domain. Prior-owner known outcomes remain queryable after restart; only
at full capacity does a new claim atomically reclaim one oldest known prior-owner
outcome to admit one command. Matching retries reclaim nothing. Pending/unknown
and current-owner valid evidence are never reclaimed under pressure.
A new served owner changes prior pending handoffs to `outcome_unknown`, without
replaying them or changing the original native run/dispatch recovery facts.
Unresolved reads use bounded keyset pages, including receipts for closed chats.


## Imported history authority (ADR-244)

The installed heads are ChaChaNotes82, Workspace9 and AgentRuns26. Their
file-backed `*_mobile_consolidation.sql` migrations add
`mobile_consolidation_batches(operation_id, source_fingerprint, batch_key,
mapping_json, committed_at)` with primary key `(operation_id, batch_key)`.
AgentRuns and Workspace also own `mobile_import_authority_holds`, keyed by
`(authority_kind, authority_id)` with an operation identity and nullable
`reviewed_at`. These schemas provide native storage; they do not perform imports.

`AgentRunsDB.hold_imported_authority(cursor, operation_id, authorities)` borrows
its active owner transaction, so imported rows, holds and batch evidence can
commit or roll back together. It requires the execution owner and refuses a
foreign cursor. Exact kinds identify native `agent_runs.id` (`run`),
`automatic_work_chains.id` (`chain`), `goal_runs.id` (`goal`),
`automatic_wake_attempts.id` (`wake_attempt`), progress message identity
(`progress`), and `automatic_chat_start_attempts.id` (`chat_start_attempt`).
Holding a chain marks it review-required without rewriting runs or accounting.
`WorkspaceDB.hold_imported_bindings(cursor, operation_id, binding_ids)` similarly
holds exact `workspace_runtime_bindings.binding_id` values. The corresponding
`imported_authority_allowed` / `imported_binding_allowed` methods read exact
holds; `DB/mobile_import_authority.py` adds linked run and allowance-family
checks for automatic consumers. There is no release API: manual work or saving
a binding never constitutes review.


The internal consolidation owner scope in `Backup_Recovery/storage_admission.py`
can construct these three exact installed repositories under native maintenance.
`participants.py` checks the actual tagged repository before core operations and
connection registration; `private_sqlite.py` accepts only the exact native
ChaChaNotes quiescent factory or the native base SQLite factory for its matching
owner/path. Retained cursors and transaction commits revalidate the live grant,
selector binding and object witnesses. Ordinary maintenance constructors and
foreign factories remain refused.

For AgentRuns, `ConsoleExecutionOwner._acquire_consolidation(scope)` accepts only
the actual live native run-owner scope. It uses the ordinary stable adjacent lock,
nonblocking flock and identity/refcount checks. After acquiring that lock, native
consolidation journals any required main/sidecar 0644-to-0600 transition before
constructing the repository. Ordinary acquisition retains its existing hardening
behavior. The native scope holds an execution use through physical SQLite
retirement; a call to `close()` while users remain cannot release the execution fence. No imported
transaction or batch receipt is manufactured by this admission machinery.

The mobile scope supplies an internal `require_private_mode` requirement to the
strict v2 private-SQLite helper request. This is derived from the real bound grant,
not a public connect flag or owner string. Every existing DB/sidecar must already
be 0600 after its journaled transition; a late nonprivate generation refuses
without mutation. The connection seam also rechecks the initial fixed cohort
after helper completion and before SQLite allocation, refusing even a private
replacement generation. The helper leaves absent optional sidecars absent.
Optional-sidecar revalidation during SQLite ownership and native 0600 main-file
creation remain unchanged. Fixed sidecar observations at ownership retirement or
recovery allow ordinary WAL generations to change without weakening permanent DB/ancestor
witnesses or writing per-row journal records.
