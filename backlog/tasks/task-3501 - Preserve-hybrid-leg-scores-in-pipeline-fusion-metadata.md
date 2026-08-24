---
id: TASK-3501
title: Preserve hybrid leg scores in pipeline fusion metadata
status: Done
assignee: []
created_date: '2026-08-07 20:35'
updated_date: '2026-08-24 17:34'
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
- [x] #1 pipeline_builder_simple.py's hybrid merge preserves each leg's original score, captured before any mutation of the shared result/item object
- [x] #2 A regression test exercises the real hybrid merge and asserts the leg scores in hybrid_fusion metadata do not silently equal the post-fusion score
- [x] #3 No behavior change for non-hybrid search modes on the legacy pipeline path
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Detailed execution plan: `Docs/superpowers/plans/2026-08-24-task-3501-pipeline-fusion-provenance.md`

1. Record and review the narrow call-site design in `Docs/superpowers/specs/2026-08-24-task-3501-pipeline-fusion-provenance-design.md`.
2. Extend the real parallel `rrf_merge` regression with distinct raw scores for overlapping, FTS-only, and vector-only rows; prove the pre-fix path fails the provenance assertions and each absent leg is recorded as `None`.
3. Snapshot each available leg score before mutating the selected result, then add those snapshots to the existing `hybrid_fusion` metadata without changing selection, ranking, citations, or non-hybrid merges.
4. Run the focused fusion suite, mutation-check the new assertion against the pre-fix behavior, review the diff, and complete task documentation.

ADR required: no

ADR path: N/A

Reason: this is a routine bug fix implementing the existing local hybrid-fusion and provenance contract in `backlog/decisions/005-invest-in-local-rag-mirroring-tldw-server.md`; it introduces no new storage, runtime, security, or cross-module boundary.

## Note (2026-08-10, TASK-4110)

The shipped hybrid fusion default `rrf_k` is now **5** (measured for chatbook's ~20-row candidate window; `RAG_Search/simplified/config.py`'s `DEFAULT_HYBRID_RRF_K`). This merge follows it automatically: TASK-4110's Task 3 gave `fusion.resolve_rrf_k` an active-profile fallback, so `_rrf_merge_parallel_results`'s `resolve_rrf_k(merge_config.get("rrf_k"))` picks up the profile value whenever the step config names none. **Both live fusion paths therefore moved together**, even though the measurement was made only on the Library path (`RAGService._hybrid_search`). That coupling is deliberate — two live fusion call sites must not disagree on one measured number — and is pinned by `Tests/RAG/test_fusion.py::TestPipelineRrfMerge::test_merge_with_no_step_rrf_k_inherits_the_shipped_profile_default`.

TASK-3501 consciously retains separate result materializers. Both paths continue to share the fusion primitive and resolved alpha/k, while their item selection, identity keys, and citation handling remain intentionally caller-specific. This task aligns only the shared score-provenance fields instead of introducing a speculative abstraction.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Updated `tldw_chatbook/RAG_Search/pipeline_builder_simple.py` to snapshot each available FTS/vector leg score before the selected shared result object is mutated with its fused score, then record those snapshots in the existing `hybrid_fusion` metadata. This preserves the existing winner selection, ranking, citation, and non-hybrid behavior.
- Extended `Tests/RAG/test_fusion.py` through the real parallel RRF merge. The RED proof raised the expected `KeyError` before implementation; the GREEN proof passed as one targeted test after implementation. Mutation witnesses also failed when the snapshot was deliberately moved after mutation: late FTS changed `media/m1` from expected `0.05` to fused `0.11`, and late vector changed `media/s2` from expected `0.1` to fused `0.77`.
- Exact task files changed are `tldw_chatbook/RAG_Search/pipeline_builder_simple.py`, `Tests/RAG/test_fusion.py`, `Docs/superpowers/specs/2026-08-24-task-3501-pipeline-fusion-provenance-design.md`, `Docs/superpowers/plans/2026-08-24-task-3501-pipeline-fusion-provenance.md`, and this task file.
- Post-rebase focused verification passed: `Tests/RAG/test_fusion.py`, `Tests/RAG/test_local_citation_capture.py`, and `Tests/RAG_Search/test_hybrid_fusion_metadata.py` reported `163 passed, 10 warnings in 4.17s`. Ruff lint passed. `git diff --check origin/dev...HEAD` and worktree `git diff --check` passed. Ruff format check retained the inherited exact baseline: `Tests/RAG/test_fusion.py` and `tldw_chatbook/RAG_Search/pipeline_builder_simple.py` would be reformatted; no formatting change was applied.
- The repository-wide local suite collected 59,680 tests and was resource-safely interrupted at about 10% after 54 completed failures; an isolated rerun reproduced all 54 failures before rebase. After rebasing onto exact `origin/dev` `6db0f9f140ff8f290c2269694d5da8b3b7c5f8cd`, the same saved 54-node set produced 53 failures and one pass on both the rebased branch and a detached exact-base worktree. The passed node was `Tests/Architecture/test_persistent_diagnostic_inventory.py::test_production_diagnostic_inventory_and_sink_topology_are_unchanged`; the two 53-node failure sets were identical, with no branch-only or base-only nodes and matching SHA-256 `8f2c9a61be6f28edb4131e8352374da2186b897df39b96c0b0e6e00baee1d58e`. The remaining failures are therefore inherited baseline failures rather than TASK-3501 regressions.
- Independent specification-compliance and code-quality reviews both approved the change with zero findings.
- ADR required: no. This routine bug fix implements the existing provenance direction in `backlog/decisions/005-invest-in-local-rag-mirroring-tldw-server.md`; it does not add a generalized materializer, dependency, schema, configuration, storage, runtime, security, or cross-module boundary.
<!-- SECTION:NOTES:END -->
