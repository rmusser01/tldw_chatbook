---
id: TASK-34390
title: Embed scoped-search query once instead of once per allowlist
status: To Do
created_date: 2026-10-11 21:44
labels:
- performance
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
RAG_Search pipeline_functions_simple.search_semantic loops over up to four source-type allowlists, and each RAGService.search re-embeds the identical query (simplified/rag_service _semantic_search) - up to 4 model forward passes or API round-trips plus sequential awaits for one user query. Embed once in search_semantic and pass the vector through to the per-allowlist store queries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A scoped semantic search performs exactly one query embedding regardless of allowlist count,Search results with the single-embed path match the previous per-allowlist results (same store queries),Existing RAG pipeline tests pass
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
