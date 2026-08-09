---
id: TASK-3994
title: Hybrid RRF fusion never merges FTS and vector legs (id-space mismatch)
status: In Progress
assignee: []
created_date: '2026-08-09 05:16'
updated_date: '2026-08-09 17:19'
labels:
  - rag
  - retrieval
  - p2
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the P1 eval harness (TASK-3894). RRF fusion matches on SearchResult.id across two mismatched id spaces: the FTS leg emits document-level ids (media_15) while the vector leg emits chunk-level ids (media_15_chunk_0). The same document can therefore never fuse into one row; it can only appear twice under two different ids. An unfused FTS row scores (1-alpha)/(rrf_k+rank), which with hybrid_alpha=0.7 is always below the worst-ranked vector row once the vector leg fills k slots, so the FTS leg contribution never survives into the fused top-k. On the P1 corpus, hybrid search returned byte-identical results to pure semantic search across all 44 golden queries (44/44 identical id-lists), crossover measured at vector rank approximately 82. In this configuration the Hybrid Basic profile is a semantic-only profile in practice.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 RRF fusion normalizes ids to a common granularity before matching legs, so a document found by both the FTS leg and the vector leg produces one fused row reflecting both contributions.
- [ ] #2 On the P1 fixture corpus, hybrid search results provably differ from pure semantic search for at least one query where the FTS leg surfaces a relevant document the vector leg alone ranks outside the top-k. (FIRST HALF MET, SECOND HALF NOT EXHIBITABLE ON THIS CORPUS - see Implementation Notes: hybrid now differs from semantic on 22 of 44 golden queries, was 0 of 44; but semantic recall@10 is already 1.000, so no relevant document sits outside the vector leg's top-k for the FTS leg to rescue. Needs a corpus addition or an AC amendment - decide in plan Task 6.)
- [ ] #3 The P1 eval harness baselines (Tests/RAG_Eval/baselines/hybrid.json) are re-stamped in the same PR as this fix, with the before and after numbers included in the PR description. (Completes in plan Task 6, same PR - deliberately not in this commit.)
- [x] #4 A regression test pins that a document found by both legs receives a fused score reflecting both contributions, not only the vector one.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-08-09-rag-port-hybrid-fusion-fixes.md (Task 4) and Docs/superpowers/specs/2026-08-09-rag-port-hybrid-fusion-fixes-design.md for the fuse-on-document-identity design; re-stamp AC completes in plan Task 6, not this fix's commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fusion now matches the two legs on document identity instead of row id.
`_fusion_doc_key` (module-level, pure, in `rag_service.py`) returns
`(source_type, source_id-or-doc_id)`, falling back to the row id when either
component is missing so metadata-less rows keep today's no-merge behaviour.
Precedence matters: the vector leg's `doc_id` is the PREFIXED document id
(`media_15`) while the keyword leg's is the bare row id (`15`), so only
`source_id`-first matching can ever line the two legs up.

Merged rows now display the VECTOR leg's item - the matched chunk, not the
whole-document FTS row. The choice is made at the call site, not in
`FusedResult.item` (left untouched for server parity and its other consumer).
That also gives merged rows the chunk metadata the downstream row mappers
read (`source_id`, `chunk_id`), so a merged row's `source_id` is the bare,
navigable id rather than `media_15`.

Both previously-never-executed paths were audited before the change:
- the citation-merge branch guarded on the DISPLAYED item having
  `citations` while dereferencing `.citations` on BOTH legs, so a mixed
  pair (one leg citation-less) would have raised `AttributeError` the first
  time it ran. Both reads are now defensive.
- its first real run merges the keyword leg's citations with the chunk's.
  On the fixture corpus that is `[] + [chunk citation]`: the keyword leg
  only emits citations when the WHOLE query string appears verbatim in the
  document, which is rare for multi-word queries, so real merged rows carry
  exactly the vector citation today. No duplication, no crash. (Possible
  follow-up: the keyword leg's phrase-exact citation extraction, the
  citation-side analogue of TASK-3995.)

Evidence (gated harness, `RAG_EVAL=1`):
- 22 of 44 golden queries now return a different id-list from semantic
  (was 0 of 44 - hybrid was byte-identical to semantic).
- A real hybrid run over media-targeted keyword queries produces genuinely
  merged rows for the first time: e.g. "Obsidian-3 lathe spindle bearing" ->
  `media_2_chunk_0`, `fts_rank=1 vec_rank=1, fts_score=0.8,
  vector_score=0.812`. Three of five probed media queries merge.
- Baseline gate PASSED (no metric outside the 0.05 band); hybrid moved:
  overall P 0.117 -> 0.105, F1 0.208 -> 0.190; keyword-category P 0.135 ->
  0.113 [warn], F1 0.236 -> 0.202 [warn]; paraphrase P 0.103 -> 0.100;
  vocabulary_mismatch P 0.105 -> 0.100; recall/MRR/NDCG unchanged at 1.000;
  mean distinct docs 9.1 -> 10.0. Precision falls BECAUSE the fix works:
  collapsing a document's chunks into one slot frees top-k slots that then
  fill with further documents, and `precision@k` divides by
  `min(k, len(retrieved))`. With semantic recall already 1.000 there is no
  headroom for the keyword leg to add a relevant document, so on this
  corpus its contribution can only show as dilution. Baselines are NOT
  re-stamped here (plan Task 6).

Modified: `tldw_chatbook/RAG_Search/simplified/rag_service.py`.
Added: `Tests/RAG_Search/test_hybrid_doc_fusion.py` (7 tests, real
producer metadata shapes). `Tests/RAG_Search/test_hybrid_fusion_metadata.py`
passes unmodified - its empty-metadata rows ride the fallback.
Mutation-checked: reverting the key to `lambda r: r.id` reds the six merge
tests and leaves the fallback pin green; flipping the display preference
back to the FTS item reds the display test (and the mixed-leg citation
test, which is display-coupled by construction).
<!-- SECTION:NOTES:END -->
