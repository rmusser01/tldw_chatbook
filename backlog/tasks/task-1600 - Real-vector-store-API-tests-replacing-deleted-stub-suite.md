---
id: TASK-1600
title: >-
  Write real vector_store.py API tests replacing the deleted stub suite
status: Done
assignee: []
created_date: '2026-07-30 23:20'
labels:
  - testing
  - rag
priority: medium
dependencies: [task-1464]
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-1464 deleted `Tests/RAG/simplified/test_vector_stores.py` (~900 lines): it imported `tldw_chatbook.RAG_Search.simplified.vector_stores` — plural, a module that never existed — caught the ImportError, defined placeholder classes inside the test file, and tested those stubs. Permanently green, testing nothing. The real module (`vector_store.py`, singular — InMemory/Chroma vector stores) deserves a small, real test file: construction, add/query/delete round-trips, persistence behavior where applicable, and the skip-gating for the chromadb extra. Keep it modest — the stub suite's 900 lines are not the bar; a few dozen real assertions beat them.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

- [x] A test file imports the REAL `vector_store` module (no fallback stubs) and fails collection if the import breaks
- [x] Round-trip coverage for the in-memory store; chromadb-backed coverage gated on the extra
- [x] Tests pass on a machine without the chromadb extra (gated skips) and with it

## Implementation Plan

1. Map the real module API (Protocol, InMemoryVectorStore, ChromaVectorStore, SearchResult)
2. In-memory: add/search round-trip, dimension-mismatch ValueError, delete_document by doc_id metadata, clear+stats, metadata_allowlist filters BEFORE top_k truncation, LRU eviction
3. Chroma (gated on the extra): round-trip + persistence across reopen on tmp_path

## Implementation Notes

`Tests/RAG/simplified/test_vector_store.py` (singular, mirroring the module):
8 tests, all passing with the chromadb extra installed; the Chroma class gates
on the extra. Real-API contracts the stub suite could never have caught, and
this authoring DID catch as corrections: `delete_document` keys on the chunk's
`doc_id` METADATA (all chunks of a document), not chunk ids; stats expose
`count`, not `document_count`. Deliberately pinned: `metadata_allowlist`
excludes out-of-scope candidates BEFORE ranking/top_k truncation (an in-scope
doc is never starved by higher-ranked out-of-scope ones — the scoping
behavior's security property). Added: `Tests/RAG/simplified/test_vector_store.py`.
