---
id: TASK-248
title: >-
  Unify the two disconnected vector-store stacks so RAG search reads what the
  embeddings stack writes
status: To Do
assignee: []
created_date: '2026-07-12 14:11'
labels:
  - rag
  - embeddings
dependencies:
  - TASK-246
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The codebase has two parallel Chroma stacks: Embeddings/Chroma_Lib.py (ChromaDBManager, used by the embeddings management surfaces and RAG_Admin local service) and RAG_Search/simplified/vector_store.py (ChromaVectorStore, which RAGService._semantic_search actually queries). They use different clients and collection conventions, so embeddings created via one path are invisible to RAG search. Converge on a single store and collection layout so every embedding created in the app is searchable. tldw_server uses one ChromaDB with per-user collections as the reference model. Filed from the 2026-07-12 RAG module audit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Embeddings created through any app path are queryable by RAG semantic search
- [ ] #2 Only one Chroma client / collection convention remains (the other stack is removed or delegates to the survivor)
- [ ] #3 A migration or documented reset path exists for pre-existing collections
- [ ] #4 Tests cover write-via-one-path read-via-search
<!-- AC:END -->
