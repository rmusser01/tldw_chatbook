---
id: TASK-860
title: >-
  Fix sql_validation.VALID_TABLES to match the real schema (keyword_collections
  is live-broken)
status: To Do
assignee: []
created_date: '2026-07-27 04:35'
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
- [ ] #1 sql_validation.VALID_TABLES['chachanotes'] includes keyword_collections and every other real table in the live schema (all 26 currently-missing names reconciled, added or deliberately excluded with a documented reason)
- [ ] #2 update_keyword_collection() succeeds end to end (create then update) without a false-positive ValueError
- [ ] #3 A test derives its expected table list from the live schema (e.g. sqlite_master or the DB's own migration-applied table set) rather than re-typing a literal list, so it catches future schema/allowlist drift
- [ ] #4 A decision is made and implemented for VALID_COLUMNS: either it fails closed for table names absent from its key set, or every real caller's table name is added to it
<!-- AC:END -->
