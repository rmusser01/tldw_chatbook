# TASK-3501 Pipeline Fusion Provenance Design

## Context

The Chat-RAG pipeline's `rrf_merge` path is live through `perform_hybrid_rag_search` and the built-in and TOML pipeline definitions. Its fusion primitive preserves both ranked input items, but `_rrf_merge_parallel_results` currently mutates the selected `SearchResult.score` to the fused score without first recording the FTS and vector leg scores. The resulting `hybrid_fusion` metadata contains ranks, RRF contributions, alpha, and k, but omits `fts_score` and `vector_score`.

The Library engine's `RAGService._fuse_hybrid_results` already implements the intended order: snapshot both leg scores, mutate the display result to the fused score, and attach the snapshots to `hybrid_fusion`.

## Goals

- Preserve the original FTS and vector scores for every pipeline hybrid result.
- Use the same `fts_score` and `vector_score` metadata vocabulary as the Library engine.
- Exercise the real parallel pipeline merge with input scores distinguishable from the fused score.
- Preserve all existing ordering, display-item selection, citation merging, alpha/k resolution, and non-hybrid behavior.

## Non-goals

- Unifying the two result-materialization functions.
- Changing reciprocal-rank-fusion math, defaults, ranking, or result identity.
- Changing pipeline definitions, retrieval functions, citations, or non-hybrid merge modes.
- Adding fields to the generic `FusedResult` value object.

## Design

For each `FusedResult` returned by `reciprocal_rank_fusion`, `_rrf_merge_parallel_results` will read `entry.fts_item.score` and `entry.vector_item.score` into local snapshots before assigning `entry.score` to the selected result. An absent leg is represented by `None`, matching the engine path.

The selected result and its metadata-merging behavior remain unchanged. The existing `hybrid_fusion` mapping gains only:

- `fts_score`: the FTS item's score before result mutation, or `None`;
- `vector_score`: the vector item's score before result mutation, or `None`.

This ordering is essential when `entry.item` aliases `entry.fts_item`: mutating the result first would also overwrite the FTS leg score.

## Intentional Separation from the Engine Materializer

The pipeline and engine paths will continue to share `reciprocal_rank_fusion`, alpha resolution, and k resolution, while retaining separate materializers. Their surrounding contracts differ:

- pipeline identity is `(source, id)`, while the engine derives keys from source-specific metadata;
- the pipeline keeps its current primary item and explicitly combines citation metadata;
- the engine selects a display item according to its own result/citation types and source rules.

Extracting a shared materializer would either require a callback-heavy abstraction or change one of those contracts. TASK-3501 therefore consciously aligns the common provenance fields at the two call sites and leaves the caller-specific behavior local.

## Data Flow and Edge Cases

1. Retrieval functions produce FTS and vector `SearchResult` lists with raw scores.
2. The pipeline interleaves same-kind sub-legs and calls the shared RRF primitive.
3. For every fused entry, both available raw scores are snapshotted.
4. The selected result receives the fused score.
5. Existing metadata and citations are merged.
6. Fusion ranks, RRF contributions, raw leg scores, alpha, and k are attached.

For an FTS-only or vector-only result, the present leg retains its original score and the absent leg is `None`. Empty inputs and non-`rrf_merge` strategies do not enter this materializer and are unchanged.

## Testing

Extend `Tests/RAG/test_fusion.py::TestPipelineRrfMerge::test_parallel_step_rrf_merge_fuses_legs`, which drives the real `_execute_parallel_step` dispatch and `rrf_merge` implementation. Give the existing overlapping, FTS-only, and vector-only items distinct raw scores and assert the exact `(fts_score, vector_score)` pair for every row shape, including `None` for an absent leg. Each top-level `score` must remain the computed RRF score and differ from its present raw input score or scores.

The test must be observed failing on the untouched implementation because the two metadata keys are absent, then passing after the minimal production change. As a discrimination check, moving either the FTS snapshot or the vector snapshot below `result.score = entry.score` must fail the corresponding overlapping/one-leg assertions. The full `Tests/RAG/test_fusion.py` suite covers fusion math, profile-default coupling, pipeline dispatch, and the engine materializer. A diff review must confirm no non-hybrid branch changed.

## ADR Check

ADR required: no

ADR path: N/A

Reason: this is a local correctness fix under the accepted hybrid-fusion direction in `backlog/decisions/005-invest-in-local-rag-mirroring-tldw-server.md`. It does not establish a new durable architecture or boundary.
