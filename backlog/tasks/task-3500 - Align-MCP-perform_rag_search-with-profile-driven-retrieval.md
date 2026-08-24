---
id: TASK-3500
title: Align MCP perform_rag_search with profile-driven retrieval
status: Done
assignee: []
created_date: '2026-08-07 20:34'
updated_date: '2026-08-24 09:12'
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
- [x] #1 MCP `perform_rag_search` resolves and honors the active RAG profile's `default_search_mode` the same way Library rag-mode search does instead of using hardcoded search behavior
- [x] #2 For the same query and active profile, MCP results use the same score-kind-aware match-strength semantics as Library rather than fabricating vector similarity
- [x] #3 Reranking-enabled profiles rerank MCP results consistently with Library, and unavailable reranking is skipped with the same disclosure instead of failing the search
- [x] #4 Existing MCP `perform_rag_search` callers continue to work without a breaking request or response API change
- [x] #5 `mcp_inspector._ScoredRow.score_kind` reflects the actual scoring path; fused and reranker scores are handled explicitly instead of blindly defaulting to `vector_similarity`
- [x] #6 Single- and multi-value media_types filters remain effective for profile-driven semantic and hybrid MCP searches
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented profile-driven MCP media retrieval through active-profile mode resolution and request-time shared runtime acquisition, exact and `$in` metadata filters, media-only allowlists, reranker degradation disclosure, shared Library score kinds/inspector behavior, and public MCP contracts. ADR: `backlog/decisions/084-mcp-profile-driven-rag-search-contract.md`.

Local GREEN evidence: consolidated suite **670 passed, 10 warnings in 135.36s**; hybrid allowlist **37 passed, 10 warnings in 9.17s**; keyword pushdown **22 passed, 10 warnings in 4.60s**; ingestion lifecycle `-k SharedRagService` **10 passed, 40 deselected, 1 warning in 0.95s** (nonzero selection); Library mode **27 passed, 1 warning in 2.48s**; metadata-filter coverage **16 passed, 1 warning in 0.68s**. The TASK-3500 style shard (Library mode, search service, inspector, active config, and RAG service) was **290 passed, 1 warning in 91.34s**. Warnings were dependency deprecations/requests compatibility; pytest also emitted sandbox-only temporary-directory cleanup messages after runs.

Static evidence is pinned to base `a84e6ba09`: `git diff --check a84e6ba09...HEAD` is green. The dynamic audit covers 21 changed Python files (20 pinned-base counterparts and one added test). Whole-file Ruff retains 14 baseline findings in HEAD (7 E702 and 7 E402), while the pinned-base snapshots have 17 (those plus 1 E402 and 2 F401); whole-file format has 12 current candidates versus 15 base candidates. The review initially found TASK-owned formatter overlap in five files, then formatted only those hunks. Post-fix line-differential evidence intersects task-added source lines from `git diff --unified=0 a84e6ba09` with current-source lines altered by `ruff format --diff` and reports **zero** remaining TASK-3500-owned overlaps.

CI, the full suite, and live-provider UAT were explicitly excluded. The existing testing-evidence lesson records the corrected lifecycle incident: snake_case selector **0/50**, exit **5**; `SharedRagService` **10 passed/40 deselected**, exit **0**. Genuine RED discrimination evidence was preserved: T1 missing helpers/delegation; T2 `$in` semantic/basic/citation misses plus nonserializable-cache crash and false metrics; T3 eager construction, missing profile/helper, runtime lifecycle/old-mode race, vacuous selector, falsey/off-thread gaps; T4 safe reason/one-result tag, construction/runtime secret leaks, wrong experiment arm, disabled-base activation and total-degraded metrics; T5 score provenance then bool/nonfinite/oversized malformed scores; T6 public copy, stale local-control descriptor, and missing schema description.

Final Task 7 scope/ADR-084 review confirms: no Library multi-source MCP route; no MCP-local service cache; no eager enhanced-runtime construction; the exact media-only `source_type=("media",)` allowlist; no fabricated keyword or vector similarity; no raw construction/runtime exception disclosure; and an unchanged public request schema with the existing seven response keys.
<!-- SECTION:NOTES:END -->
