---
id: TASK-985
title: MCP search_conversations and RAG chunk resource call APIs that do not exist
status: Done
assignee:
  - '@claude'
created_date: '2026-07-27 20:05'
labels:
  - mcp
  - bug
  - notes-followup
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-983's whole-module pass over MCP/server.py's collaborator modules (tools.py, resources.py, prompts.py) found two more call sites of the same shape as TASK-854/968/983, deliberately left unfixed because resolving them needs a design call rather than a mechanical rename. MCPTools.search_conversations (backing server.py's search_conversations tool) calls self.chachanotes_db.search_all_content(search_query=, content_type=, limit=), a method that does not exist anywhere in the codebase at all, so the tool cannot work. The nearest real equivalent, CharactersRAGDB.search_conversations_by_content(search_query, limit), returns conversation rows with no inline content column, so the tool's current preview formatting (result["content"][:200]) cannot be ported as-is -- what the preview should be sourced from instead is a product decision. MCPResources.get_rag_chunk_resource (backing the rag-chunk://{chunk_id} MCP resource) calls self.media_db.get_chunk_by_id(int(chunk_id)), a method that also does not exist; the nearest real accessor, the module-level get_chunk_text(db_instance, chunk_uuid), takes a UUID string rather than an integer id and returns only bare chunk text, with no media_id/start_char/end_char/embedding_id metadata (UnvectorizedMediaChunks has no embedding_id column at all) -- so both the resource's id scheme and its promised metadata need to be reconciled with what the database can actually provide, not guessed at.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 MCPTools.search_conversations calls a real CharactersRAGDB method with arguments that actually exist and its formatted preview field is sourced from real data per a deliberate decision (not guessed)
- [x] #2 MCPResources.get_rag_chunk_resource calls a real MediaDatabase accessor with an identifier scheme and returned metadata fields that are reconciled with what the database can actually provide
- [x] #3 A test exercises both fixed tool/resource paths end to end against a temp database rather than only importing the module
<!-- AC:END -->

## Implementation Plan

1. Read `MCPTools.search_conversations` (tools.py), `MCPResources.get_rag_chunk_resource`
   (resources.py), and the real DB APIs (`CharactersRAGDB.search_conversations_by_content`,
   `.search_messages_by_content`, `Client_Media_DB_v2.get_chunk_text`, and the
   `UnvectorizedMediaChunks`/`conversations`/`messages` table schemas) to see exactly what
   columns/arguments genuinely exist.
2. Decide, per call site, whether to adapt (rewire to real data with an honest schema) or
   remove (if nothing useful can be returned).
3. Implement the fix, updating docstrings/schemas so the tool never promises a field it
   cannot supply.
4. Add end-to-end tests against a real temp on-disk SQLite database (no mocks), seeding rows
   directly for tables with no public writer that returns a known id/uuid.
5. Run `Tests/MCP/` plus the touched data-layer module's own test suite.

## Implementation Notes

**search_conversations (adapt).** `search_all_content` never existed on `CharactersRAGDB`.
Rewired to `search_conversations_by_content(search_query=, limit=)`, the real FTS accessor,
which returns `conversations` rows (real columns: `id`, `title`, `created_at`, `character_id`,
plus an aggregated `message_count`) but no content column at all — there is nothing on the
conversation row itself to preview. Rather than fabricate a preview from the title, `preview`
is now sourced from a second, deliberate query per matching conversation:
`search_messages_by_content(content_query=query, conversation_id=..., limit=1)` — the actual
best-matching *message* text in that conversation, against the same FTS index the
conversation-level search already matched on. This is real seeded data, not a guess, and
keeps the same truncate-to-200-chars convention `search_notes` already uses elsewhere in this
module. Caller-visible contract: `search_conversations` still returns
id/title/preview/created/character_id/message_count; `preview` is now "the matching message
text" rather than "conversation content" (there is no such thing), documented in both
`MCPTools.search_conversations`'s docstring and the `server.py` tool's docstring.

**get_rag_chunk_resource (adapt).** `get_chunk_by_id` never existed on `MediaDatabase`. The
nearest real accessor, `get_chunk_text(db_instance, chunk_uuid)`, keys on a UUID (not an int
id) but returns only the bare `chunk_text` string — no `media_id`/`start_char`/`end_char`, and
`UnvectorizedMediaChunks` has no `embedding_id` column at all (chunks are correlated to the
vector store by this same UUID, not a separate id). Rather than either invent an `embedding_id`
or throw away the position/media metadata the resource used to promise, added a small sibling
accessor, `get_chunk_by_uuid(db_instance, chunk_uuid)` (`Client_Media_DB_v2.py`, next to
`get_chunk_text`), that selects the chunk's real columns
(`id`/`uuid`/`media_id`/`chunk_text`/`chunk_index`/`start_char`/`end_char`/`chunk_type`) by
UUID with the same active-chunk/active-media join `get_chunk_text` already uses. Reworked
`get_rag_chunk_resource` (and the `rag-chunk://{chunk_uuid}` MCP resource route in `server.py`)
to take `chunk_uuid` instead of an int `chunk_id`, matching how chunks are actually addressed
elsewhere in the app. `embedding_id` is dropped from the returned metadata (documented as
nonexistent); `chunk_index`/`chunk_type` are added since they are real, previously-unreported
columns.

**Tests.** Added 4 tests to `Tests/MCP/test_tools_resources_prompts_real_methods.py`
(TASK-983's sibling-defect test module) against real on-disk SQLite databases, no mocks:
`test_search_conversations_uses_a_real_content_search_accessor`,
`test_search_conversations_filters_by_character_id`,
`test_get_rag_chunk_resource_uses_a_real_media_db_accessor`, and
`test_get_rag_chunk_resource_reports_not_found_for_unknown_uuid`. A `_seed_chunk` helper
inserts an `UnvectorizedMediaChunks` row directly with a known UUID, mirroring the existing
`_seed_transcript` helper's rationale: `process_unvectorized_chunks` (the public writer)
generates its own UUID internally and never returns it, so there is no public way to seed a
chunk with a UUID a test can assert against afterwards.

**Verification:** `Tests/MCP/` (381 passed), `Tests/Media_DB/test_media_db_v2.py` +
`Tests/DB/test_search_conversations_fts.py` (54 passed, covering the touched
`Client_Media_DB_v2.py` module and the pre-existing `search_conversations_by_content` DB
method this fix now calls from MCP).

**Files changed:** `tldw_chatbook/MCP/tools.py`, `tldw_chatbook/MCP/resources.py`,
`tldw_chatbook/MCP/server.py`, `tldw_chatbook/DB/Client_Media_DB_v2.py` (new
`get_chunk_by_uuid` function, additive), `Tests/MCP/test_tools_resources_prompts_real_methods.py`.
