---
id: TASK-3601
title: Library plain-profile rag mode should not require the embeddings runtime
status: To Do
assignee: []
created_date: '2026-08-07 22:06'
labels:
  - rag
  - library
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Final-review finding on the P0 branch: _search_rag resolves the heavy RAG runtime before reading the profile mode, so BM25 Only + missing embeddings_rag extras renders "RAG unavailable -- install embeddings" although the plain route needs no vectors, and with deps present the first plain query pays embedding-model construction to read a flag; resolve_active_rag_config() can supply the mode without a runtime.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With BM25 Only active and embeddings_rag absent, Library rag mode runs the four-seam keyword search instead of the unavailable state
- [ ] #2 No embedding construction on the plain route
<!-- AC:END -->
