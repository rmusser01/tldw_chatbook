---
id: TASK-3601
title: Library plain-profile rag mode should not require the embeddings runtime
status: Done
assignee:
  - '@claude'
created_date: '2026-08-07 22:06'
updated_date: '2026-11-05 23:06'
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
- [x] #1 With BM25 Only active and embeddings_rag absent, Library rag mode runs the four-seam keyword search instead of the unavailable state
- [x] #2 No embedding construction on the plain route
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed by TASK-33620.15.41 (branch perf/task-33620.15.41-bm25-skips-embedding-runtime). LibraryLocalRagSearchService reads the active profile's search mode and name without a runtime (active_config.resolve_active_rag_search_mode, the new resolve_active_rag_profile_name), off the event loop and once per shared-service generation. For a keyword-only (plain) profile, _search_rag and prewarm run the four-seam keyword search and build nothing, so no embedding model is constructed and the embeddings_rag gate is never consulted. Any other mode, a cached runtime, or an unreadable profile resolves the runtime exactly as before. The Library screen's rag mode calls the same service (run_library_rag_search -> app.library_rag_search_service), so it gets the same route. Tests: Tests/Library/test_library_rag_plain_profile_without_runtime.py, red on dev 29592ecc9b for AC#1 (RAG unavailable without the extras) and AC#2 (one runtime build; a fresh interpreter imports torch, transformers, sentence_transformers and chromadb), green on the branch, with controls for an embedding profile and an unreadable profile. Docs: Docs/User_Guide/library/search-and-rag.md.
<!-- SECTION:NOTES:END -->
