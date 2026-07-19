---
id: TASK-260
title: RAG conversation search: batch fetch + skip image BLOBs
status: Done
assignee: ['@claude']
created_date: '2026-07-16 14:30'
labels: [performance, rag]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
pipeline_functions_simple.search_conversations_fts5 (96-115) issues one get_messages_for_conversation per matched conversation (≤20 queries/search) and every call SELECTs image_data BLOBs only to build text snippets. get_messages_for_conversations_batch (ChaChaNotes_DB 5870-5929) exists unused. Add an include_image_data=False variant for text-only callers (also mindmap_integration.py:88). Full analysis with measurements: Docs/Design/2026-07-16-performance-audit.md (§P3 D2).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 One batched query replaces the per-conversation loop
- [x] #2 Text-only callers no longer fetch BLOB columns
- [x] #3 RAG search results unchanged (tests)
<!-- AC:END -->

## Implementation Plan
<!-- SECTION:PLAN:BEGIN -->
1. Probe the live path end-to-end against a real DB before touching it (unmocked), then wire get_messages_for_conversations_batch into search_conversations_fts5.
2. Add include_image_data (default True) to both fetch methods; pass False from the pipeline and mindmap text-only callers.
3. Equivalence + BLOB-skip + end-to-end tests on a real file-backed CharactersRAGDB.
<!-- SECTION:PLAN:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
The pre-work probe found the function had NEVER worked end-to-end — three latent defects, all invisible to mocked tests: `from ...DB.ChaChaNotes_DB` resolves beyond the top-level package (leftover from the simplified/ flattening; raises ImportError at call time), and the ctor call was missing the required client_id. Both fixed here (the lines were being rewritten anyway); `search_notes_fts5` has the same class of breakage plus a nonexistent target module — filed as task-295 rather than widened scope. In production none of this crashed because nothing sets app.db_config, so the function short-circuited to [] — that wiring decision is also part of task-295.

The batch rewrite: one get_messages_for_conversations_batch call (include_image_data=False) replaces the per-conversation loop; results are assembled by iterating the relevance-ordered conv_results and looking up the batch dict, so content and order are byte-identical to the old loop (test-asserted against a reference loop on a seeded real DB). BLOB skip: include_image_data=False swaps the SELECT column for NULL AS image_data — key present, value None, image_mime_type still returned so callers can tell an image exists (dict-shape-stable choice; consumers do .get()-style reads). Mindmap's text-only fetch (Tools/Mind_Map/mindmap_integration.py — the task file's RAG_Search path was stale) also passes include_image_data=False.

Tests: Tests/RAG_Search/test_conversation_search_batch.py (4, all real-DB unmocked). Suites: ChaChaNotesDB + chat persistence/conversation/functions + Chatbooks 385 passed; 3 failures reproduced identically at the base commit (pre-existing legacy-parity drift, listed in PR).

Files: DB/ChaChaNotes_DB.py, RAG_Search/pipeline_functions_simple.py, Tools/Mind_Map/mindmap_integration.py, Tests/RAG_Search/test_conversation_search_batch.py, backlog/tasks/task-295 (new bug filing).
<!-- SECTION:NOTES:END -->
