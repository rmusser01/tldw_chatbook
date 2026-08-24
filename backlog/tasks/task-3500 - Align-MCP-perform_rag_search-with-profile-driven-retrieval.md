---
id: TASK-3500
title: Align MCP perform_rag_search with profile-driven retrieval
status: In Progress
assignee: []
created_date: '2026-08-07 20:34'
updated_date: '2026-08-24 05:55'
labels:
  - rag
  - mcp
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The RAG-port P0 programme (TASK-3170) made Library rag-mode search honor the active RAG profile's plain/semantic/hybrid `default_search_mode`, score-kind-aware match bands, reranking, and rebuilt keyword leg. MCP `perform_rag_search` still has a genuine parity gap with that profile-driven behavior, so the same query can produce different retrieval and match-strength semantics through MCP and Library.

This task aligns only MCP `perform_rag_search`. The legacy agent-owner premise is already satisfied and superseded: `LibraryRagToolProvider.search_library_rag` owns fallback RAG retrieval and `LibraryToolProvider.library_search_notes` owns direct note retrieval under ADR-030. Those agent paths are no longer in scope here.
<!-- SECTION:DESCRIPTION:END -->

## Architecture Decision

- [ADR-084: MCP profile-driven RAG search contract](../decisions/084-mcp-profile-driven-rag-search-contract.md)

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 MCP `perform_rag_search` resolves and honors the active RAG profile's `default_search_mode` the same way Library rag-mode search does instead of using hardcoded search behavior
- [ ] #2 For the same query and active profile, MCP results use the same score-kind-aware match-strength semantics as Library rather than fabricating vector similarity
- [ ] #3 Reranking-enabled profiles rerank MCP results consistently with Library, and unavailable reranking is skipped with the same disclosure instead of failing the search
- [ ] #4 Existing MCP `perform_rag_search` callers continue to work without a breaking request or response API change
- [ ] #5 `mcp_inspector._ScoredRow.score_kind` reflects the actual scoring path; fused and reranker scores are handled explicitly instead of blindly defaulting to `vector_similarity`
- [ ] #6 Single- and multi-value media_types filters remain effective for profile-driven semantic and hybrid MCP searches
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Implement the shared active-profile search-mode resolver and Library normalization.
2. Add exact and single-key membership metadata filtering across semantic and keyword engine post-filters.
3. Make the MCP media adapter lazy, profile-driven, shared-runtime-backed, and media-confined while preserving the boolean API and response shape.
4. Make shared reranker construction degradation credential-safe, disclosed, and reset-clean.
5. Interpret MCP scores through the existing Library score-kind vocabulary.
6. Update and pin MCP public documentation.
7. Run focused local verification and record TDD discrimination evidence before closeout.

Detailed plan: Docs/superpowers/plans/2026-08-23-task-3500-mcp-profile-driven-rag-search.md

ADR required: yes
ADR path: backlog/decisions/084-mcp-profile-driven-rag-search-contract.md
Reason: TASK-3500 changes the lasting MCP request/runtime contract.
<!-- SECTION:PLAN:END -->
