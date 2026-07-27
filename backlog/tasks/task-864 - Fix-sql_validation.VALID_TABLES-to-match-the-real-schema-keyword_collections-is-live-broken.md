---
id: TASK-864
title: >-
  Fix sql_validation.VALID_TABLES to match the real schema (keyword_collections
  is live-broken)
status: Done
assignee:
  - '@claude'
created_date: '2026-07-27 04:35'
updated_date: '2026-07-27 17:28'
labels:
  - security
  - db
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
DB/sql_validation.py:14-24's VALID_TABLES['chachanotes'] allowlists 9 table names; the ChaChaNotes schema actually has 35 tables. This is a live, reproducible feature break, not just a theoretical gap: the allowlist includes the link table collection_keywords but omits the entity table keyword_collections, and ChaChaNotes_DB.py:9309 update_keyword_collection() -> _update_generic_item() -> :4312 validate_table_name(table_name, "chachanotes") raises ValueError for it. A direct reproduction -- add_keyword_collection('Coll A') followed by update_keyword_collection(1, {'name': 'Coll B'}, expected_version=1) -- created the collection successfully and then failed the update with "Invalid table name: keyword_collections". A full diff against the live schema found 25 more real tables missing from the allowlist (character_expression_images, chat_dictionaries, conversation_dictionaries, conversation_local_marks, conversation_world_books, db_schema_version, decks, flashcard_assets, flashcard_templates, flashcards, learning_paths, message_attachments, message_generation_metadata, mindmap_nodes, mindmaps, quiz_attempts, quiz_questions, quizzes, review_history, study_sessions, sync_conflicts, sync_sessions, topics, world_book_entries, world_books).

Separately, DB/sql_validation.py:308's VALID_COLUMNS gate (if table_name and table_name in VALID_COLUMNS) silently no-ops for any table_name not among its 8 keys. Real call sites pass table names it doesn't recognize -- "sync_profile_state" (Sync_Interop/sync_state_repository.py:1875, immediately before an ALTER TABLE ... ADD COLUMN f-string) and Transcripts/MediaChunks/UnvectorizedMediaChunks/DocumentVersions (DB/Client_Media_DB_v2.py:2950-2952, :3180-3182) -- so only the generic \w+/reserved-word filter applies to them, not the column-specific check their surrounding code comments claim is in effect. Not exploitable today since all of these inputs are in-file literals, but the schema validation those call sites document is not actually delivered.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 sql_validation.VALID_TABLES['chachanotes'] includes keyword_collections and every other real table in the live schema (all 26 currently-missing names reconciled, added or deliberately excluded with a documented reason)
- [x] #2 update_keyword_collection() succeeds end to end (create then update) without a false-positive ValueError
- [x] #3 A test derives its expected table list from the live schema (e.g. sqlite_master or the DB's own migration-applied table set) rather than re-typing a literal list, so it catches future schema/allowlist drift
- [x] #4 A decision is made and implemented for VALID_COLUMNS: either it fails closed for table names absent from its key set, or every real caller's table name is added to it
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the update_keyword_collection break against origin/dev (add_keyword_collection then update_keyword_collection) and capture the ValueError.
2. Instantiate a real, fully-migrated CharactersRAGDB(":memory:") and read sqlite_master to get the true, current table set (excluding FTS5 shadow/virtual tables and sqlite_sequence); diff it against VALID_TABLES['chachanotes'].
3. Reconcile VALID_TABLES['chachanotes'] by hand against that derived list (documenting why derivation-at-runtime is not done inside sql_validation.py itself: circular-import/heavyweight-DB-construction concerns), and add a test that re-derives the live set and fails on any future divergence in either direction.
4. Re-run the reproduction to confirm update_keyword_collection now succeeds; add regression tests for add/update/soft-delete of keyword_collections (currently zero coverage).
5. Enumerate every real caller of validate_column_name(column, table_name) across the codebase to see which table names are passed; decide whether to fail closed for tables missing from VALID_COLUMNS or backfill every real caller's table -- do both: add the missing tables (keyword_collections, sync_profile_state, Transcripts, MediaChunks, UnvectorizedMediaChunks, DocumentVersions) so nothing currently working regresses, then make the fallback fail closed.
6. Run the sql_validation, ChaChaNotesDB, and related DB test suites to confirm no regressions.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reproduced the break first (see report): add_keyword_collection('Coll A') then update_keyword_collection(1, {'name': 'Coll B'}, expected_version=1) raised ValueError: Invalid table name: keyword_collections on origin/dev. Confirmed fixed after this change.

Reconciled VALID_TABLES['chachanotes']: instantiated a real, fully-migrated CharactersRAGDB(":memory:") and read sqlite_master directly rather than trusting the task's own 26-name list, which turned out to already be stale -- schema v27 (RAG citation provenance, merged since the audit) added 12 more rag_* tables the audit never saw. The live schema has 47 substantive tables (excluding FTS5 shadow/virtual tables and sqlite_sequence); all 47 are now allowlisted (38 were missing: the 26 the audit found + 12 newer rag_* tables + keyword_collections itself was already counted in the 26).

Did NOT derive VALID_TABLES at runtime/import time from a live DB: doing so would mean sql_validation.py -- a lightweight, dependency-free identifier validator shared by three otherwise-unrelated DB modules (chachanotes/media/prompts) -- constructing a full CharactersRAGDB (running all 27 migrations, ~30 log lines) as a side effect of validating a table name, and would require a lazy in-function import back into ChaChaNotes_DB.py to avoid a circular import. Instead: the allowlist stays hand-maintained (with a comment explaining why), and Tests/DB/test_sql_validation.py::TestChachanotesValidTablesMatchesLiveSchema derives the real table set the same way (sqlite_master on a live migrated DB) and fails in both directions (missing OR stale) the moment they diverge -- this is what actually "catches future schema/allowlist drift" per AC #3, since a hand-copied list (like the task's own 26 names) demonstrably goes stale.

VALID_COLUMNS decision (AC #4): chose BOTH options together rather than either alone. Enumerated every real call site of validate_column_name(column, table_name) across the repo: the only tables ever passed with a concrete table_name are character_cards/conversations/messages/notes/keywords (already present), keyword_collections (ChaChaNotes_DB.update_keyword_collection/soft_delete_keyword_collection), sync_profile_state (Sync_Interop/sync_state_repository._ensure_sync_v2_profile_columns, right before an ALTER TABLE ADD COLUMN f-string), and Transcripts/MediaChunks/UnvectorizedMediaChunks/DocumentVersions (Client_Media_DB_v2's soft-delete/undelete cascade loops). Added VALID_COLUMNS entries for all of these (columns verified against each table's real CREATE TABLE / PRAGMA table_info), then changed validate_column_name's fallback from "silently skip the check" to "return False" for any table_name that still isn't registered. Rationale: failing closed with nothing backfilled would have repeated the exact 864 pattern (an absent entry unconditionally breaking a currently-working call site); backfilling without failing closed leaves the exact silent-no-op gap the audit flagged. Two currently-dead, fully-dynamic helper methods (Prompts_DB._get_next_version, Client_Media_DB_v2._get_next_version) have no live callers today, so they were left as-is -- if ever revived for a table without a VALID_COLUMNS entry, fail-closed is the correct (safe) behavior for them too, not a regression since nothing calls them now.

Modified/added files:
- tldw_chatbook/DB/sql_validation.py (VALID_TABLES['chachanotes'] reconciled to 47 real tables with rationale comment; VALID_COLUMNS gained keyword_collections/sync_profile_state/Transcripts/MediaChunks/UnvectorizedMediaChunks/DocumentVersions; validate_column_name fails closed for unregistered tables)
- Tests/DB/test_sql_validation.py (added keyword_collections to the existing valid-tables test; added TestChachanotesValidTablesMatchesLiveSchema deriving the expected table set from sqlite_master on a live migrated DB, both directions)
- Tests/ChaChaNotesDB/test_chachanotes_db.py (added TestKeywordCollections: add+update and add+soft-delete lifecycle tests -- this table had zero test coverage before, which is how the omission went unnoticed)
<!-- SECTION:NOTES:END -->
