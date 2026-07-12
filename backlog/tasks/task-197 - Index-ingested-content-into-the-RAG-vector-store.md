---
id: TASK-197
title: Index ingested content into the RAG vector store
status: To Do
assignee: []
created_date: '2026-07-12 14:11'
labels:
  - rag
  - embeddings
  - ingest
dependencies:
  - TASK-196
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Nothing in the app ever populates the RAG vector store: the only indexing entry point (index_documents_modular in Event_Handlers/Chat_Events/chat_rag_integration.py:300, which calls the non-existent `rag_service.embed_documents` method) has zero callers, and no app code invokes index_document/index_batch on any RAG service. Semantic and hybrid search therefore query an empty store and the vector leg always contributes nothing. Mirror the tldw_server design where chunking plus embedding plus storage happens at ingestion time via a background worker so semantic search has something to search. Filed from the 2026-07-12 RAG module audit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Newly ingested media is chunked and embedded and indexed into the RAG vector store when embeddings deps are installed via a non-blocking background worker
- [ ] #2 A semantic search for distinctive content of a newly ingested document returns that document
- [ ] #3 A bulk backfill path exists to index pre-existing media/notes/conversations
- [ ] #4 Indexing failures are logged and surfaced without breaking ingestion
- [ ] #5 When embeddings deps are missing no indexing is attempted and ingestion is unaffected
<!-- AC:END -->
