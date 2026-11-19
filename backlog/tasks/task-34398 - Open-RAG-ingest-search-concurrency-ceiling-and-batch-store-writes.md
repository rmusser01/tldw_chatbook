---
id: TASK-34398
title: Open RAG ingest/search concurrency ceiling and batch store writes
status: Done
created_date: 2026-10-12 03:20
labels:
- performance
priority: medium
assignee:
- '@zcode'
updated_date: 2026-10-12 04:47
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Embeddings/Embeddings_Lib.py EmbeddingFactory.embed (lines ~781-831) holds one global lock across the entire model forward pass, so background IngestionIndexer batches fully serialize interactive query embeddings (concurrency 1 regardless of hardware). RAG_Search/simplified/indexing_helpers.py store_documents_batch (~278) issues one vector-store add + executor hop + Chroma commit per document instead of one per batch. ChromaVectorStore.add (vector_store.py ~371) converts the whole (m,d) numpy matrix to Python lists although chromadb accepts arrays. RAG_Search/simplified/embeddings_wrapper.py create_embeddings (~479-509) md5/sha256-hashes every text to probe a cache keyed by model id - always misses, ~100% miss telemetry.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 EmbeddingFactory guards only cache mutation with the lock; model work runs outside it with a refcount/epoch so in-use models are not evicted; concurrent embed + ingest no longer serialize,store_documents_batch issues one vector_store.add per batch (interfaces already accept lists),ChromaVectorStore.add passes the ndarray through without tolist(),The always-miss cache probe in embeddings_wrapper is removed or replaced with a real content-hash cache,Existing embeddings/RAG tests pass and a concurrency test shows two overlapping embed calls interleave
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Lock-scope reduction in EmbeddingFactory with refcount-protected eviction 2. batch store_documents_batch 3. chroma ndarray passthrough 4. remove always-miss probe 5. concurrency test + notes
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented and verified (897 consolidated tests green); see task file Implementation Notes; PR #3083
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
