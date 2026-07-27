---
id: TASK-985
title: MCP search_conversations and RAG chunk resource call APIs that do not exist
status: To Do
assignee: []
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
- [ ] #1 MCPTools.search_conversations calls a real CharactersRAGDB method with arguments that actually exist and its formatted preview field is sourced from real data per a deliberate decision (not guessed)
- [ ] #2 MCPResources.get_rag_chunk_resource calls a real MediaDatabase accessor with an identifier scheme and returned metadata fields that are reconciled with what the database can actually provide
- [ ] #3 A test exercises both fixed tool/resource paths end to end against a temp database rather than only importing the module
<!-- AC:END -->
