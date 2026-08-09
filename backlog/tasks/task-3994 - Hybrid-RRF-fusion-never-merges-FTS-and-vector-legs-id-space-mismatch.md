---
id: TASK-3994
title: Hybrid RRF fusion never merges FTS and vector legs (id-space mismatch)
status: To Do
assignee: []
created_date: '2026-08-09 05:16'
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
- [ ] #1 RRF fusion normalizes ids to a common granularity before matching legs, so a document found by both the FTS leg and the vector leg produces one fused row reflecting both contributions.
- [ ] #2 On the P1 fixture corpus, hybrid search results provably differ from pure semantic search for at least one query where the FTS leg surfaces a relevant document the vector leg alone ranks outside the top-k.
- [ ] #3 The P1 eval harness baselines (Tests/RAG_Eval/baselines/hybrid.json) are re-stamped in the same PR as this fix, with the before and after numbers included in the PR description.
- [ ] #4 A regression test pins that a document found by both legs receives a fused score reflecting both contributions, not only the vector one.
<!-- AC:END -->
