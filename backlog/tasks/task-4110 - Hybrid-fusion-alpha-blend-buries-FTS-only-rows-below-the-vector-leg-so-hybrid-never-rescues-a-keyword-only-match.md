---
id: TASK-4110
title: >-
  Hybrid fusion alpha blend buries FTS-only rows below the vector leg, so hybrid
  never rescues a keyword-only match
status: To Do
assignee: []
created_date: '2026-08-09 20:22'
updated_date: '2026-08-09 20:59'
labels:
  - rag
  - retrieval
  - p2
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found closing the TASK-3994/3995/3996 fusion cluster (plan Task 6), by the fixture the cluster added for TASK-3994 AC #2. TASK-3994 fixed the id-space half of its own description (the two legs now fuse on document identity and merged rows carry both contributions). The other half of that description was never addressed: an FTS-only row - a document the keyword leg found and the vector leg did not return at all - still cannot enter hybrid's fused top-k. The fused score is (1-alpha)*1/(rrf_k+fts_rank) + alpha*1/(rrf_k+vector_rank), so with the shipped defaults (hybrid_alpha 0.7, rrf_k 60) an FTS-only row at keyword rank 1 scores 0.3/61 = 0.00492 and is beaten by every vector row ranked better than about 82, while _hybrid_search only ever asks the vector leg for top_k * SEARCH_RESULT_MULTIPLIER (2) results. Whenever the vector leg returns k or more DISTINCT DOCUMENTS - the normal case - the keyword leg's unique finds are structurally unreachable, which is the sense in which Hybrid Basic is still a semantic-only profile for documents the vector leg misses. Distinct documents rather than rows is load-bearing: fusion dedups by document identity, so a vector leg whose 2k chunk rows collapse to fewer than k documents leaves slots free and keyword-only rows do appear - that is why the live check saw them on a profile whose vector index held media chunks only while the query was scoped to notes and conversations. Do not read those sightings as evidence the starvation is absent. Measured on the P1 fixture corpus with the new golden query 'plant maintenance record' (note-saltmarsh-hide): plain returns it at rank 1, semantic does not return it at all, the engine's FTS leg returns it at rank 1, and hybrid does not return it - it sorts 21st behind 20 vector rows. Feeding the same fused function a larger vector candidate pool made the target merge at vector rank 22 and land at fused rank 1, which shows the burial is a scoring/weighting property and not a wiring bug. The fix is a retrieval-design decision (renormalise per-leg RRF, give the keyword leg a slot quota, retune alpha/rrf_k, or widen the candidate pool) and needs its own before/after baseline re-stamp under the Tests/RAG_Eval P2 discipline - do not change the shared RAG_Search/fusion.py blend without one, and note that pipeline_builder consumes the same function (TASK-3501).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A document that only the keyword leg finds can appear in hybrid's top-k when the vector leg returns k or more distinct documents (the case that is structurally impossible today - a thin vector leg already lets keyword-only rows through and is not evidence)
- [ ] #2 The chosen weighting is justified against a measurement, not asserted: the before/after numbers for the affected golden queries appear in the PR
- [ ] #3 Tests/RAG_Eval baselines are re-stamped in the same PR, and the hybrid cell for kw-plant-maintenance-record moves from miss to hit
- [ ] #4 A regression test pins that an FTS-only row outranks at least one vector-only row under the shipped default alpha and rrf_k
- [ ] #5 A test asserts the vector-blind fixture is STILL vector-blind - semantic mode does not return note-saltmarsh-hide for kw-plant-maintenance-record - so a future model bump or re-stamp cannot silently return the corpus to the state where it cannot distinguish coverage from noise
<!-- AC:END -->
