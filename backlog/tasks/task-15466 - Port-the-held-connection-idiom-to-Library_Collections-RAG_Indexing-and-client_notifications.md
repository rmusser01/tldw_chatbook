---
id: TASK-15466
title: >-
  Port the held-connection idiom to Library_Collections, RAG_Indexing, and
  client_notifications
status: Done
assignee:
  - '@claude'
created_date: '2026-08-11 12:05'
updated_date: '2026-08-11 22:09'
labels:
  - perf
  - db
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
From the audit: three wired DBs still open a fresh sqlite connection per operation — the pre-task-3011 anti-pattern. `DB/Library_Collections_DB.py:26-43` (`closing()` per op; its service also opens write-`BEGIN` transactions for pure reads at `Library/library_collections_service.py:380/:450/:519`; hot on the Library screen, though off-loop via to_thread); `DB/RAG_Indexing_DB.py:102/:136` (fresh connection PER INDEXED ITEM in `mark_item_indexed`, and `with self._get_connection()` is a transaction context manager so connections are never closed, only GC'd — plus a per-item fsync at DELETE/FULL); `Notifications/client_notifications_db.py:56+` (same never-closed shape, insert per dispatched notification). Each fresh connect costs ~2x a raw connect because the private-sqlite seam re-verifies the DB file and three sidecars per open (`DB/private_sqlite.py:948-991`).

Fix direction: the Workspace_DB held thread-local idiom — WITH `isolation_level=None` (the task-3012 lesson: without it, sqlite3 auto-BEGINs on DML and the accumulated implicit transaction breaks explicit BEGIN and silently rolls back bare DML on close). Mandatory pre-port step per that lesson: audit every `connection()` call site for bare DML and record the audit in this task. Pure-read paths stop opening write transactions. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All three DBs hold thread-local connections with isolation_level=None; the pre-port DML call-site audit is recorded in this task
- [x] #2 Pure-read service calls no longer open write transactions (evidence)
- [x] #3 RAG indexing no longer opens/commits per item — batched (evidence)
- [x] #4 All three DBs' existing test surfaces green; no connection leak (probe or unit evidence)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Pre-port DML audit: enumerate every connection()/_get_connection()/cursor call site in Library_Collections_DB, RAG_Indexing_DB, client_notifications_db (+ their service consumers) and classify each read / single-statement write / multi-statement write, recording the audit in the task notes (task-3012 lesson: bare DML on a held autocommit connection loses implicit-transaction atomicity; an explicit BEGIN inside an accumulated implicit txn raises).
2. Pin behaviour: run the existing suites green BEFORE any edit (Library collections service + local library tool service + MCP library tools, RAG indexing DB + ingestion indexing, client notifications DB/service/dispatch/inbox, pragma pairing, sqlite owner-privacy/inventory, home-dashboard seams, parity state).
3. Port Library_Collections_DB to the Workspace_DB held thread-local idiom: thread-local conn + idle liveness ping + isolation_level=None, WAL+NORMAL re-applied per new connection, transaction() -> BEGIN IMMEDIATE on the held conn, plus a new read_transaction() (BEGIN DEFERRED + rollback) so pure reads keep their multi-statement snapshot without taking the write API.
4. Move library_collections_service's three pure-read methods (list_library_collections, search_library_collections, get_library_collection) off db.transaction() onto db.read_transaction().
5. Port RAG_Indexing_DB the same way (it is not a BaseDB subclass): held thread-local connection, connection()/transaction() context managers replacing 'with self._get_connection()', explicit transaction for the multi-statement clear_all, and a batched mark_items_indexed() so the ingestion loop marks a whole batch in ONE transaction instead of one open+commit per item; update the task-15465 WAL-pinning comment to point at this fix.
6. Point RAG_Search/ingestion_indexing.py's per-item mark loop at the batch API, preserving best-effort semantics (tracking failure warns; it never changes the indexed count).
7. Port client_notifications_db the same way, keeping the existing single shared :memory: connection (thread-local would give each thread its own empty DB) and wrapping the multi-statement update_settings loop in an explicit transaction.
8. Evidence: add a connection-count probe test (one connection per thread, not per op, across all three DBs), a test that the pure-read service calls never enter the write transaction, and a test that a batch of N indexed items produces ONE transaction; verify MCP/server.py constructor compatibility.
9. Re-run every pinned suite green; record the audit + evidence in Implementation Notes; tick ACs; commit code + task file.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Ported the three remaining fresh-connection-per-operation DBs to the Workspace_DB held thread-local idiom, with isolation_level=None per the task-3012 lesson. Each fresh open cost ~2x a raw sqlite3.connect because the private-SQLite seam re-verifies the DB file plus three sidecars every time; the count is now bounded by the number of threads that touch the DB rather than by call volume.

## Pre-port DML call-site audit (mandatory, task-3012)

Every connection()/_get_connection() site in the three modules and their consumers, classified. R = read-only, W1 = single-statement write (safe under autocommit -- each statement is its own transaction), Wn = multi-statement write (needs an EXPLICIT transaction under autocommit, else the statements commit independently), DDL = executescript (self-commits under either isolation mode).

Library_Collections_DB.py + Library/library_collections_service.py
- _initialize_schema (DDL) -- executescript; the trailing conn.commit() was already a no-op and is unchanged.
- get_schema_version (R), service list_collections :164 (R), get_collection :190 (R), _ensure_unique_name :603 (R) -- all via connection().
- service create_collection :225 (W1), rename_collection :269 (W1), delete_collection :295 (W1), add_item_to_collection :325 (W1) -- all already via transaction(); kept there.
- service list_library_collections :380, search_library_collections :450, get_library_collection :519 -- PURE READS that were calling the write API. Each pairs a COUNT with its page and genuinely needs one snapshot, so they moved to a new read_transaction() (BEGIN DEFERRED + always ROLLBACK), not to a bare connection: dropping to autocommit would have silently split count and page across two snapshots. transaction() itself is now BEGIN IMMEDIATE, so the write path is unambiguous.
- No nesting anywhere: _ensure_unique_name / get_collection are called BEFORE the transaction block, never inside one.

RAG_Indexing_DB.py
- _initialize_schema (DDL), mark_item_indexed (W1), remove_indexed_item (W1, reads cursor.rowcount), update_collection_state (W1), get_indexed_item_info / get_indexed_items_by_type / get_collection_state / get_indexing_stats (R).
- clear_all -- Wn (two DELETEs). Was relying on the implicit transaction; now wrapped in an explicit transaction() so the wipe stays atomic.
- New mark_items_indexed -- Wn by construction (executemany in one transaction).
- Note: 'with self._get_connection() as conn' was sqlite3's TRANSACTION context manager, never a closing one, so the old connections leaked until GC.

Notifications/client_notifications_db.py
- _initialize_schema (DDL), insert_notification (W1, reads lastrowid), _update_flags (W1, reads rowcount), get_notification / list_notifications / list_notifications_after_id / get_settings (R).
- update_settings -- Wn (one upsert per key). Now wrapped in an explicit transaction() so a multi-key update stays all-or-nothing.

## Decisions / trade-offs

- read_transaction() is new API rather than reusing connection(): AC#2 asks pure reads to stop opening write transactions, but the three seams document 'count and page are read in one transaction' and that guarantee is load-bearing. BEGIN DEFERRED takes no lock until a statement needs one and is always ended with ROLLBACK, so it satisfies both.
- transaction() upgraded from BEGIN to BEGIN IMMEDIATE on Library_Collections: MCP/server.py opens this same file in its own process, so a deferred transaction that read before writing could fail to upgrade.
- client_notifications keeps its single shared :memory: connection (an in-memory DB lives inside its connection; thread-local would hand each thread an empty inbox). Home/active_work_adapter.py's existing is_memory_db guard remains correct and untouched.
- No busy_timeout pragma added: sqlite3.connect defaults to timeout=5.0 (busy_timeout 5000ms) and the private seam does not override it, so the AgentRuns_DB busy_timeout line would be redundant here.
- Batching stops at mark_item_indexed. remove_indexed_item stays per-item because remove_entries adjusts its summary per removal; it no longer opens a connection per item either way.
- Ingestion tracking stays best-effort: a failed batch write warns and does NOT decrement 'indexed' (the documents are indexed; untracked ones are re-indexed next run), matching the per-item form it replaces.
- MCP/server.py:390 constructs LibraryCollectionsDB(path, client_id) -- signature unchanged; Tests/MCP/test_library_tools.py's signature test passes.

## Evidence

New Tests/DB/test_held_connections.py (25 tests) measures the properties rather than the code:
- Connection count: after construction, 12 service reads + 2 writes / 25 marked items / 20 inserted notifications open ZERO new connections; 4 threads x 2 ops open exactly 4.
- task-3012 trap: bare DML survives db.close() (would be silently rolled back without isolation_level=None); an explicit BEGIN after bare DML does not raise; isolation_level is None on all three held connections.
- AC#2: SQL captured via set_trace_callback shows the three agent read seams emit exactly one BEGIN DEFERRED + one ROLLBACK and NEVER BEGIN IMMEDIATE/COMMIT, while create_collection emits exactly one BEGIN IMMEDIATE + COMMIT.
- AC#3: a 50-item batch emits exactly one BEGIN IMMEDIATE + COMMIT; a failing batch leaves zero rows (partial tracking would silently skip un-indexed items forever); index_entries() calls mark_items_indexed ONCE and mark_item_indexed zero times for a 12-entry batch.
- Mutation-checked (Edit-based restore): removing isolation_level=None reddens 4 tests; reverting one read seam to transaction() reddens the matching read test.

Suites green: 240 passed (held-connection + Library collections + local library tool + MCP library tools + RAG indexing DB + pragma pairing + notifications DB/service/dispatch/inbox + parity state), 289 passed (RAG ingestion + sqlite owner-privacy/inventory/interop-owners + home-dashboard seams + quiz/research interop + watchlists services + briefing presets), 146 passed (watchlists destination shell + Notifications). Whole-suite --collect-only clean: 37,829 collected, no import errors. ruff clean on all changed files.

## Files

Modified: tldw_chatbook/DB/Library_Collections_DB.py, tldw_chatbook/DB/RAG_Indexing_DB.py, tldw_chatbook/Notifications/client_notifications_db.py, tldw_chatbook/Library/library_collections_service.py, tldw_chatbook/RAG_Search/ingestion_indexing.py, Tests/DB/test_pragma_settings.py (helpers now read pragmas off the connection the DB actually uses).
Added: Tests/DB/test_held_connections.py.

## Fix round 1 (review: Approved with 1 Important + 5 minors)

1. IMPORTANT -- `read_transaction()` silently discarded writes. It always ends in ROLLBACK, so DML placed inside it vanished with no error and no log. It now compares `conn.total_changes` across the block and, after rolling back, raises RuntimeError naming the misuse. Deliberately scoped: `total_changes` cannot see a DML that matched zero rows or bare DDL, so this is a guard against silent DATA LOSS, not general read-only enforcement -- stated in the docstring rather than implied. Only Library_Collections has a `read_transaction()`; RAG_Indexing and client_notifications read through `connection()` (no transaction, nothing to discard), so there is no shared helper to apply it to.
2. `read_transaction`/`transaction` docstrings in all three DBs now state that nesting on one thread raises `OperationalError: cannot start a transaction within a transaction` (pre-port each block had its own connection, so nesting silently "worked") and that the outer block still rolls back cleanly.
3. RAG_Indexing `:memory:` asymmetry documented rather than special-cased -- uniform thread-local matches the Workspace_DB template it was ported from, and production always uses a file path. The comment states the consequence outright (a second thread gets a schema-less DB) and that it is not a regression (pre-port EVERY call opened a fresh empty in-memory DB).
4. `update_settings()` returns early when passed no settings, so a zero-statement BEGIN IMMEDIATE no longer takes the write lock on what is effectively a read (`update_preferences` reaches it that way).
5. Pragma test gained `test_worker_thread_connection_also_pairs_wal_and_synchronous_normal` across 5 held-connection DBs, reading the pragmas off a WORKER thread's own connection -- `synchronous` is per-connection, and the construction-thread factories could not tell "applied once" from "applied per new connection".
6. Removed the leftover no-op `conn.commit()` after `executescript` in `_initialize_schema`.

Fix-round evidence (both new claims mutation-checked, Edit-based restore): disabling the write guard reddens the new write-inside-read test; making the pragma apply only to the FIRST connection reddens ONLY the new worker-thread test while the pre-existing construction-thread assertion stays green -- i.e. the new test catches exactly what the old suite could not. Re-runs: 314 passed (held-connections + pragma + Library collections + local library tool + MCP library tools + RAG indexing DB + notifications DB/service/dispatch/inbox + Tests/Notifications + parity state), 261 passed (RAG ingestion + sqlite owner-privacy/inventory/interop-owners + home-dashboard seams + quiz/research interop + watchlists services). ruff clean.
<!-- SECTION:NOTES:END -->
