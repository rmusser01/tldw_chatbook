---
id: TASK-3501
title: >-
  Preserve hybrid leg scores in pipeline fusion metadata
status: In Progress
assignee: []
created_date: '2026-08-07 20:35'
updated_date: '2026-08-24 15:25'
labels:
  - rag
dependencies:
  - TASK-3170
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The live Chat-RAG pipeline merge in `RAG_Search/pipeline_builder_simple.py` replaces the selected result's raw score with its fused score but does not record either input leg's original score in `hybrid_fusion`. The Library engine path in `RAG_Search/simplified/rag_service.py` already snapshots and records `fts_score` and `vector_score` before that mutation. Callers routed through the pipeline builder therefore lose score provenance even though ranks and RRF contributions remain available.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 pipeline_builder_simple.py's hybrid merge preserves each leg's original score, captured before any mutation of the shared result/item object
- [ ] #2 A regression test exercises the real hybrid merge and asserts the leg scores in hybrid_fusion metadata do not silently equal the post-fusion score
- [ ] #3 No behavior change for non-hybrid search modes on the legacy pipeline path
<!-- AC:END -->

## Implementation Plan

1. Record and review the narrow call-site design in `Docs/superpowers/specs/2026-08-24-task-3501-pipeline-fusion-provenance-design.md`.
2. Extend the real parallel `rrf_merge` regression with distinct FTS and vector scores and prove the pre-fix path fails the provenance assertions.
3. Snapshot each available leg score before mutating the selected result, then add those snapshots to the existing `hybrid_fusion` metadata without changing selection, ranking, citations, or non-hybrid merges.
4. Run the focused fusion suite, mutation-check the new assertion against the pre-fix behavior, review the diff, and complete task documentation.

ADR required: no

ADR path: N/A

Reason: this is a routine bug fix implementing the existing local hybrid-fusion and provenance contract in `backlog/decisions/005-invest-in-local-rag-mirroring-tldw-server.md`; it introduces no new storage, runtime, security, or cross-module boundary.

## Note (2026-08-10, TASK-4110)

The shipped hybrid fusion default `rrf_k` is now **5** (measured for chatbook's ~20-row candidate window; `RAG_Search/simplified/config.py`'s `DEFAULT_HYBRID_RRF_K`). This merge follows it automatically: TASK-4110's Task 3 gave `fusion.resolve_rrf_k` an active-profile fallback, so `_rrf_merge_parallel_results`'s `resolve_rrf_k(merge_config.get("rrf_k"))` picks up the profile value whenever the step config names none. **Both live fusion paths therefore moved together**, even though the measurement was made only on the Library path (`RAGService._hybrid_search`). That coupling is deliberate — two live fusion call sites must not disagree on one measured number — and is pinned by `Tests/RAG/test_fusion.py::TestPipelineRrfMerge::test_merge_with_no_step_rrf_k_inherits_the_shipped_profile_default`.

TASK-3501 consciously retains separate result materializers. Both paths continue to share the fusion primitive and resolved alpha/k, while their item selection, identity keys, and citation handling remain intentionally caller-specific. This task aligns only the shared score-provenance fields instead of introducing a speculative abstraction.
