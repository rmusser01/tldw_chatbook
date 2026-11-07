# Database layer

This document describes the SQLite layer: the `BaseDB` infrastructure (private connections, quiescence, semantic-mutation guards), the ChaChaNotes schema and its file-backed migrations, the Media DB and its in-code migrations, the cross-DB patterns (transactions, optimistic locking, soft deletion, FTS5), and the satellite databases.

## Authoritative files

| File | Role |
| --- | --- |
| `DB/base_db.py` | `BaseDB` ABC (path handling, `':memory:'`, client ids, private connections, vacuum, integrity check) plus process-wide machinery: `SQLiteConnectionQuiescenceRegistry`, quiescent connection/cursor wrappers, `_SemanticMutationAuthorization` + `register_semantic_mutation_guard` |
| `DB/ChaChaNotes_DB.py` | `CharactersRAGDB` — conversations, messages, characters, notes, keywords, world books, learning tables, console policy columns. Schema version is defined by `_CURRENT_SCHEMA_VERSION` |
| `DB/Client_Media_DB_v2.py` | `MediaDatabase` — media items, chunks, transcripts, document versions, chunking templates. Schema v9 |
| `DB/migrations/chachanotes_vN_to_vN+1_*.sql` | File-backed ChaChaNotes migration steps (v16→v67+ present) |
| `DB/store_kernel.py` | `SQLiteStoreKernel` mixin — the one held-connection/transaction implementation (ADR-353): autocommit held thread-local connections, 30s liveness probe, BEGIN IMMEDIATE `transaction()` with BaseException rollback, participants wiring. `RAG_Indexing_DB`, `Workspace_DB`, and `Guardian_DB` consume it; Guardian preserves admission rechecks around native connection setup; remaining stores migrate one per PR |
| `DB/sql_validation.py` | Identifier validation (`escape_identifier`, `validate_table_name`, `validate_column_name`) |
| `DB/Prompts_DB.py`, `Evals_DB.py`, `Subscriptions_DB.py`, `AgentRuns_DB.py`, `Library_*_DB.py`, `RAG_Indexing_DB.py`, `Workspace_DB.py`, `VisualIdentity_DB.py`, `Chunking_Lab_DB.py` | Satellite databases |

Note: the "schema v37" figure that circulates in older docs is stale — read `_CURRENT_SCHEMA_VERSION` in `DB/ChaChaNotes_DB.py` for the current version.

## BaseDB infrastructure

- **Private connections** (`connect_private_sqlite`) harden the SQLite build for user-data files.
- **Finite Workspace reads**: registry `get_workspace`, `list_workspaces` and `get_runtime_binding` use `operation_owned_connection` to retire newly acquired exact file-backed handles on their producer thread. The binding read includes Default-binding repair; borrowed handles/transactions and custom/in-memory owners keep their lifetimes. Awaiter cancellation does not retire a running callback (ADR-353).
- **Quiescence registry**: an exclusive maintenance barrier per canonical file identity. Cursors hold use reservations until result exhaustion; `quiesce_connections()` drains and closes same-file handles so maintenance (vacuum, migration) can run without "database is locked" races.
- **Semantic-mutation guards**: `register_semantic_mutation_guard` registers SQLite functions plus a trace callback and an authorizer that **denies COMMIT/ROLLBACK escape mid-scope** — fail-closed guards consumed by DB triggers protecting console trace/semantic state.

## ChaChaNotes (CharactersRAGDB)

### Schema

Core tables: `character_cards`, `conversations` (TEXT UUID PK, `root_id`, `forked_from_message_id`, character FK), `messages` (with `parent_message_id` for the branch tree), `keywords` / `keyword_collections` + link tables, `notes` (with folders and organization sync ids), `research_quick_note_owner_proofs` (local-only), `sync_log` (trigger-populated change feed for server sync), plus later additions: `world_books`, learning/flashcard tables, `message_attachments`, `message_generation_metadata`, canvas revisions, and the console policy/memory columns (`console_project_context_json`, `console_conversation_memories`, `active_leaf_message_id`).

### Migrations

Steps are `.sql` files under `DB/migrations/`, executed **statement-by-statement through a cursor inside the step transaction** (an earlier `executescript` approach auto-committed and was replaced). Idempotent-replay guards handle re-runs: require-entry-version checks, skip-already-applied `ADD COLUMN`, drop-superseded-trigger. The runner is kept aligned with the folder by test.

Schema 82 (ADR-239) permits tied Canvas sibling sequences without rewriting any
immutable rows. The file-backed 81→82 transaction verifies the exact predecessor
index, replaces its uniqueness with `(canvas_id, sequence, id)` lookup ordering,
adds a unique partial root index, and guards the version update. Fresh installs
replay the same step. Recovery retains every frozen predecessor catalog, all four
v81 lineages and their exact v82 successors, including installed shared-file
Subscriptions catalogs; only those exact catalogs can enable the pure Canvas
payload CHECK during bounded validation. Restricted connections relinquish that
temporary authority before returning.

Schema85 appends local conversation `archived_at` chronology and its archive lookup index after the existing schema83 portrait and schema84 region migrations. Existing immutable Canvas and semantic columns remain unchanged; real archive/restore transitions advance local conversation versions while archive-only writes emit no sync event or payload field. Workspace schema10 follows the schema9 binding-authority migration and adds local `archived_at` and `archive_revision`, preserving unknown historical dates and compare-and-swap restoration. Archive search uses finite operation deadlines through existing BaseDB/SQLiteStoreKernel connection custody, including borrowed transactions.

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

- `Prompts_DB` — prompts + keywords, own migrations to v6, Python-side FTS, soft deletes, expected-version conflicts, and Prompt/Recipe/Mix-in discriminator. The v5-to-v6 file-backed rebuild preserves IDs, sequence, references, indexes, FTS and retained sync history; recovery accepts exact v4/v5 catalogs and migrates staged copies under its restricted authorizer.
- `Evals_DB` — eval tasks/models/runs/results + FTS; version tracked via `PRAGMA user_version` (no version table).
- `Subscriptions_DB` — watchlist/subscription state + briefing provenance; FTS-with-LIKE-fallback at the search boundary ("the search box must never raise into the reader").
- `AgentRuns_DB` — agent run/step records, automatic-work accounting, private Goal history, and native workflow admission/history (schema v30).
- `Library_Collections_DB` / `Library_Ingest_Jobs_DB` — collections and the durable ingest job queue.
- `RAG_Indexing_DB` — incremental indexing state (see [rag.md](./rag.md)).
- `Workspace_DB`, `VisualIdentity_DB`, `Chunking_Lab_DB` — workspace bindings, visual identity, chunking experiments.

`Agents/workflow_models.py` and `DB/workflow_runs.py` implement the passive portion of [ADR-285](../../backlog/decisions/285-native-phased-agent-workflows.md). Workflow launches and their automatic chains are created together through the existing connection-taking FULL transaction. Immutable request/plan records, normalized phases/tasks, native run/attempt links, decisions, active-interval rows and required results stay in AgentRunsDB; separate control revisions and progress cursors prevent result publication from acting as approval. The passive history store never starts a coordinator, dispatch, startup audit or resume capability.

Plans are limited to 128 KiB, six phases, twelve task definitions and 64 footprint entries per task. Required result slots are reserved before effects, capped at 128 KiB per record, 4 MiB per workflow and 128 MiB across workflow payloads (including retained request/plan/decision bytes). Publication checks exact workflow/attempt/native run ownership; missing mandatory results fail explicitly. Settled result removal retains immutable plans, accounting and reservation tombstones; active, paused, uncertain or physically undrained records cannot be removed. The passive retirement field records an existing native owner's observation and never retires an ExecutionOwner itself.

Stopped, completed, closed and removed workflows have closed control revisions. The existing FULL transaction checks for competing unfinished workflows in the same chat. Original native attempts can still finish required publication and physical retirement. Admission and restricted archive validation charge an eligible settled result/checkpoint pair at its exact stored UTF-8 byte length. The original 128 KiB reservation stays intact for replay and oversized-pair rejection. A missing checkpoint, pending retirement or unsettled allowance family retains its original charge. The current chat’s status view can remove one selected workflow’s result/checkpoint payloads and Change Review window details after confirmation. Immutable history, original accounting and native Change Review snapshots keep their retention, so capacity can still be exhausted.

The shipped `agent_runs_v24_to_v25_review_boundaries.sql` retains review-boundary claims. Workflow migrations follow it: v25→v26 introduces passive history, v26→v27 adds planning/execution admission and active intervals, and v28–v30 add checkpoints/resume commands, change windows and immutable repair-instruction digests. Exact frozen local workflow v25–v29 catalogs have a narrow transactional bridge to the combined lineage; an unrecognized hybrid is refused before constructor DDL. The same classification applies to staged restore, and migrated history grants no execution authority. The installed `DB/recovery_operations.py` owner, frozen `DB/recovery_workflow_schema.py` catalog and narrow `Backup_Recovery/sqlite_validation.py` staging allowlist qualify actual constructor and migration SQL without rewriting historical catalogs. The installed catalog data freezes every exact review-boundary v25 and canonical v26–v30 result plus the independently qualified historical v30 producers. The development generator uses the private memory-only `recovery.workflow_catalog` owner. Opening an existing file compares complete physical objects, implicit indexes, table/index metadata and version stamps before constructor DDL. The three historical v30 producers require every stamp from 4 through 30. A failed optional receipt savepoint is recognized only for the original current and local v29 producers and that local producer’s v30 bridge, with all other stamps present. Only the ordinary constructor retries it. SQLite planner statistics qualify as empty, stat1 only, or stat1 and stat4 with their exact installed metadata. Statistics may contain private samples and travel inside the database backup. Live operational validation reads committed WAL through its admission-aware owner in one deferred transaction. Staged migration permits only SQLite’s measured statistics deletes during the installed v26→27 phase-table drop. Validators check payload hashes, plan structural rows, reservation capacity and cross-owner references without opening captured binding paths. Recovery contract imports remain constructor-free; restored history and saved decisions grant no execution authority and remain subject to existing recovered-content activation.

`DB/automatic_work.py` now admits `workflow_phase` attempts using the same FULL
transaction and resource reservations as Goal/fleet work. Planning has a nullable
phase reference and explicit purpose, not a fabricated executable plan. Execution
preparation reserves mandatory task-result capacity; task claims expose one native
child reservation and stable requested run ID. Native launch consumes that same
reservation once. Duplicate decision IDs return their original reference; changed
payloads and revoked control revisions refuse.

Workflow active time uses one open interval per workflow, charged through physical
cleanup uncertainty. Positive drain plus durable required results closes an interval;
planning publication and closure commit atomically. Within-phase quiescence closes
only a positively drained interval, retaining acceptance and unlaunched claims;
explicit resume and native provider/child dispatch reopen it transactionally. Planning
failed/cancelled settlement retains statuses `failed`/`closed`, consumed charges and
control revision without publishing a plan; only confirmed drain releases unused
result capacity. Unknown usage refuses closure. Drained review/capacity waits consume no
time. Foreign open intervals retain uncertainty and conservatively exhaust the
captured allowance at startup. When every unknown native row is already
`review_required`, passive accounting returns unknown before importing Console.
Mixed and accepted rows retain the original live-witness evaluation. Goal/fleet absolute deadlines are unchanged. The
phase coordinator must supply the native physical-drain observation; a logical
terminal row or user acknowledgment is insufficient.

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

The scheduled sync-log retention pass invokes each original DB callback through
`Backup_Recovery.participants.run_finite_local_worker` on its existing worker.
The helper closes newly acquired installed file-backed caches through the
original owner after callback return. Preexisting/foreign caches, transactions,
custom/in-memory owners and uncertain retirement keep their existing custody.

## Related docs

- [chat-pipeline.md](./chat-pipeline.md) — the messages/conversations consumers
- [notes-sync.md](./notes-sync.md) — the notes tables' sync authority
- [media-ingestion.md](./media-ingestion.md) — the Media DB write seam


### Review-boundary wake linkage (ADR281)

AgentRuns schema25 extends existing `automatic_wake_attempts` with a nullable opaque
`review_boundary_id` and `automatic_review_boundary_claims`. `AutomaticWorkLedger`
validates an in-process host receipt at claim and acceptance; persisted identities
cannot restore that authority. Prepared abort removes its exact active link and
reservation, while accepted/completed/uncertain attempts retain links and charges.
The existing restart audit requires review and cannot recover ephemeral bundles.
Source, report and grant bodies are never stored in this link.

## Operational Console prompt mix-ins (ADR-286)

ChaChaNotes v86 installs `chachanotes_prompt_mixins.sql`. The caller-cursor
repository `DB/console_prompt_mixins.py` stores local conversation preferences,
immutable used definitions, attempts and ordered attempt items in four
`console_prompt_mixin_*` tables. Preferences have their own CAS revision; corrupt
or future formats are preserved and unavailable to ordinary saves. Definitions
are bounded by the existing Prompt limits. A digest narrows reuse candidates,
but complete source/version/name/placement/body equality decides reuse.

Attempts retain exact instructions independently of diagnostic capture. Explicit
zero-item records mean recorded-empty; missing marked records, damaged item counts
or invalid snapshots mean unavailable. Body-free `mixin_attempt_id` and
`mixin_snapshot_schema_version` facts live in assistant metadata; an update trigger
preserves them through terminal projections. Active dispatch checkpoints also
carry the nullable schema marker. Legacy unmarked attempts remain distinguishable.

`MixinAttemptContribution` uses the existing transaction writer's narrow
`append_mixin_attempt` capability. Fresh durable acceptance persists staged native
IDs unchanged and passes an exact identity map; promotion passes its existing
native-to-durable map. Legacy contributions retain their separate role aliases.
The contribution's frozen values participate in durable acceptance fingerprints;
bodies are never generic metadata or log fields.

Fork transactions copy each referenced snapshot once into destination ownership,
remap anchors and preserve historical retry markers without copying preferences.
Message deletion nulls anchors; conversation hard purge cascades records. The v86
migration also fixes `conversations_ad` to avoid deleting an already soft-deleted
FTS posting twice. Historical v81–v85 catalogs remain unchanged.

Qualification uses disposable real databases for v85→v86 candidate migration, capture-off restore, temporary promotion, copy/fork and source purge; source candidate bytes remain unchanged during restore validation. The installed-distribution contract checks SQL migration inclusion in both wheel and source archive. These snapshots retain exact instruction content even when diagnostic capture is off: trace deletion alone does not delete operational Retry content. See [QA evidence and limits](../superpowers/qa/2026-11-01-prompt-mix-ins/verification.md).

The migration statement parser and leading-comment scanner live in
`DB/sql_validation.py`; ChaChaNotes supplies its original schema error type.
The public `Prompts_DB.export_prompts_formatted` delegates lazily to
`DB/prompt_export.py`, keeping current DB dependency bindings and legacy export
formats. A filtered selection containing Mix-ins is refused before output creation;
the typed Library/Chatbook export preserves their canonical definition and placement.
Sealed citation row-set comparison lives in
`Chat/citation_artifact_ownership.py` and uses the repository's already acquired
connection; transaction and seal ownership remain on the repository.


### PR3411 integrated definition owners — Task13, 2026-11-07

The bounded catalog/metadata implementations and canonical helper limits belong to `DB.recovery_sqlite`. Backup keeps restrictions, installed-reference/validation orchestration and its live local helper/pre-statement slots; DB/workflow readers explicitly use the DB slots. Old Backup aliases are the same function objects, not synchronized mutable limit/patch registries. Existing caller-owned transactions, WAL view, FTS restriction reset, catalog/statistics/stamp rules, limits and errors are unchanged.

| Names | Producer / old entry / actual defining globals | Receiving/capture ownership |
| --- | --- | --- |
| `_catalog`, `_metadata`, two helper limits | DB common helper owns executing bodies/globals; Backup names are aliases | Backup local calls/prechecks remain live Backup slots; DB/workflow sites use DB slots. Same passed restrictions and caller-owned transaction. |
| Media SQL tuple | New no-import data leaf; old service alias before class; no function moves | Original service global lookup/transaction, original class/descriptors/full collectors. No record issued by leaf. |
| Personal SQL constants/tuple | Existing models data; old repository aliases | Original repository code/globals/schema/encryption/transaction/capture remain owners. |

The Media leaf contains only the existing SQL tuple and adds one real project module at the local service import phase. Preserve exact SQL bytes/order, old aliases, existing database owners and every original cold cap; it adds no schema service, connection, migration, source record or permission. Personal declaration relocation changes no schema policy/version or encryption/transaction ownership. Governance: ADR-285, with ADR-029/126 custody and ADR-097/230 cold constraints.
