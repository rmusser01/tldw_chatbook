---
id: TASK-291
title: >-
  Semantic and hybrid pipeline search must not silently return empty when the
  RAG service is uninitialized
status: To Do
assignee: []
created_date: '2026-07-12 14:11'
labels:
  - rag
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
search_semantic returns an empty list whenever app._rag_service is unset (RAG_Search/pipeline_functions_simple.py:184-186, logged only as a warning). The standalone SearchRAGWindow never initializes the service, so its contextual mode always returns nothing and hybrid quietly degrades to FTS-only; the chat sidebar initializes it lazily but other callers do not. Users cannot distinguish no-matches from not-initialized from missing-deps. The Library canvas already surfaces honest recovery states for this (library_local_rag_search_service.py); the chat and standalone Search surfaces should reach the same bar. Filed from the 2026-07-12 RAG module audit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Selecting semantic or contextual or hybrid mode in chat or standalone Search either initializes the RAG service or visibly reports why semantic retrieval is unavailable (missing deps or uninitialized runtime or empty index)
- [ ] #2 Hybrid results clearly indicate when they are FTS-only because the vector leg was unavailable
- [ ] #3 No code path remains where a user-triggered semantic search silently returns an empty result due to an uninitialized service
<!-- AC:END -->
