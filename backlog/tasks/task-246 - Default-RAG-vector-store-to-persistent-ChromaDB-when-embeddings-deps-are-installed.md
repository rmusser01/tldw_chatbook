---
id: TASK-246
title: >-
  Default RAG vector store to persistent ChromaDB when embeddings deps are
  installed
status: To Do
assignee: []
created_date: '2026-07-12 14:11'
labels:
  - rag
  - embeddings
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The RAG vector store defaults to type memory (RAG_Search/simplified/config.py:48) and the hybrid_basic runtime profile does not override it, so the semantic index starts empty on every launch and nothing can persist across sessions. tldw_server uses persistent ChromaDB as its default store. When the embeddings_rag extra is installed, the chatbook RAG service should default to a persistent ChromaDB store under the user data directory so indexed content survives restarts. Filed from the 2026-07-12 RAG module audit (see backlog/docs/rag-module-audit-2026-07-12.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With embeddings_rag installed the RAG service vector store persists across app restarts (ChromaDB with a persist directory under the user data dir)
- [ ] #2 Without embeddings deps behavior is unchanged: plain FTS5 search works and no import errors occur
- [ ] #3 Config still allows an explicit override back to the in-memory store
- [ ] #4 Store selection logic is covered by tests for both with-deps and without-deps cases
<!-- AC:END -->
