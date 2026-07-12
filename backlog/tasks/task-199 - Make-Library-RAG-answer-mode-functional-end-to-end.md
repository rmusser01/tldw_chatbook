---
id: TASK-199
title: Make Library RAG-answer mode functional end-to-end
status: To Do
assignee: []
created_date: '2026-07-12 14:11'
labels:
  - rag
  - library
dependencies:
  - TASK-197
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Library Search canvas is wired to LibraryLocalRagSearchService (app.py:3128): search mode runs real FTS5 over the notes/media/conversations seams and works today. But rag mode (RAG Answer) delegates to app._rag_service (library_local_rag_search_service.py:239), which is only ever created by the chat sidebar path (chat_rag_events.py get_or_initialize_rag_service, its sole caller is chat_rag_events.py:339). The Library screen never initializes it, so RAG Answer always returns the RAG-unavailable recovery state; and even when a prior chat semantic search created the service, the vector index is empty (see task-197) so it returns zero rows. Initialize the RAG runtime from the Library path when embeddings deps are present and return real semantic results. Filed from the 2026-07-12 RAG module audit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Selecting RAG Answer mode with embeddings deps installed initializes the RAG runtime instead of returning the RAG-unavailable recovery state
- [ ] #2 With indexed content present a RAG Answer query returns semantic results with citations in the evidence rows
- [ ] #3 Without embeddings deps the existing recovery state still routes the user to setup
- [ ] #4 When the RAG runtime is available but the index is empty the outcome says so instead of showing a bare zero-results state
<!-- AC:END -->
