---
id: TASK-3501
title: >-
  pipeline_builder_simple.py hybrid merge has the same leg-score aliasing bug
  Task 2 fixed in rag_service
status: To Do
assignee: []
created_date: '2026-08-07 20:35'
labels:
  - rag
dependencies:
  - TASK-3170
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-3170's Task 2 fixed a bug in RAG_Search/simplified/rag_service.py's hybrid fusion where per-leg (keyword/vector) scores were captured AFTER result.score had already been mutated, so entry.item aliased fts_item and the preserved 'original' leg scores were actually already-fused values. RAG_Search/simplified/pipeline_builder_simple.py (~L360-386) is the legacy pipeline-builder path and has its own, separate hybrid merge with the identical aliasing pattern -- it was out of scope for Task 2 (which only touched rag_service.py) and was never fixed. Any caller still routed through the legacy pipeline builder gets the same corrupted per-leg scores in hybrid_fusion metadata that rag_service.py used to produce before Task 2.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 pipeline_builder_simple.py's hybrid merge preserves each leg's original score, captured before any mutation of the shared result/item object
- [ ] #2 A regression test exercises the real hybrid merge and asserts the leg scores in hybrid_fusion metadata do not silently equal the post-fusion score
- [ ] #3 No behavior change for non-hybrid search modes on the legacy pipeline path
<!-- AC:END -->

## Note (2026-08-10, TASK-4110)

The shipped hybrid fusion default `rrf_k` is now **5** (measured for chatbook's ~20-row candidate window; `RAG_Search/simplified/config.py`'s `DEFAULT_HYBRID_RRF_K`). This merge follows it automatically: TASK-4110's Task 3 gave `fusion.resolve_rrf_k` an active-profile fallback, so `_rrf_merge_parallel_results`'s `resolve_rrf_k(merge_config.get("rrf_k"))` picks up the profile value whenever the step config names none. **Both live fusion paths therefore moved together**, even though the measurement was made only on the Library path (`RAGService._hybrid_search`). That coupling is deliberate — two live fusion call sites must not disagree on one measured number — and is pinned by `Tests/RAG/test_fusion.py::TestPipelineRrfMerge::test_merge_with_no_step_rrf_k_inherits_the_shipped_profile_default`. When this task lands, unify or consciously diverge the two paths rather than letting the coupling stay implicit.
