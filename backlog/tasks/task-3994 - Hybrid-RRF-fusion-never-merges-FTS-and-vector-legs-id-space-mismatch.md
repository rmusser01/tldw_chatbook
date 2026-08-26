---
id: TASK-3994
title: Hybrid RRF fusion never merges FTS and vector legs (id-space mismatch)
status: Done
assignee: []
created_date: '2026-08-09 05:16'
updated_date: '2026-08-09 20:40'
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
- [x] #2a MERGE HALF - MET. AMENDED IN PLAN TASK 6: as originally written, AC #2 bundled two claims under one checkbox, so a single tick could not tell the truth about both; it is split here into #2a and #2b so the board shows what actually landed. The first claim - hybrid results provably differ from pure semantic on the P1 corpus - is MET: 22 of 44 golden queries returned a different id-list immediately after the fix, was 0 of 44 (hybrid had been byte-identical to semantic).
- [x] #2b RESCUE HALF - CLOSED 2026-08-10 BY TASK-4110's MEASURED WEIGHTING (see the closure note at the end of this task's Implementation Notes for the numbers). Original text, unedited, follows. NOT MET BY THIS FIX, delegated to TASK-4110. The second claim - the FTS leg surfacing a relevant document the vector leg ranks outside the top-k - is not delivered here, and Task 6 established why by building the missing evidence rather than assuming it. The corpus had no vector-blind document (semantic recall@10 was 1.000 everywhere), so one was authored: note-saltmarsh-hide / kw-plant-maintenance-record, now committed. Measured against it: plain returns it at rank 1, semantic does not return it at all (the fixture works), the engine's FTS leg returns it at rank 1 - and hybrid still does not return it, because fusion's alpha blend scores an FTS-only row at (1-alpha)/(rrf_k+1) = 0.00492 and the vector leg fills every slot above it. That is a SECOND defect, named in this task's own description but not addressed by the id-space fix, and it is filed with its before-number as TASK-4110. This box stays UNTICKED on purpose: TASK-4110 closes it, and the harness will show it as kw-plant-maintenance-record's hybrid cell going from miss to hit. Leaving it ticked would have let a bundled checkbox certify work no commit on this branch performed.
- [x] #3 The P1 eval harness baselines (Tests/RAG_Eval/baselines/hybrid.json) are re-stamped in the same PR as this fix, with the before and after numbers included in the PR description. (Completes in plan Task 6, same PR - deliberately not in this commit.)
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
  exactly the vector citation today. No duplication, no crash. (That
  phrase-exact citation extraction -- the citation-side analogue of
  TASK-3995 -- was closed later in the same PR: `_keyword_citation_spans`
  now locates spans from the SAME token list the MATCH expression is built
  from, so a scattered-token hit still cites.)

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

**Plan Task 6 closure (re-stamp + AC #2 adjudication).** Baselines re-stamped
once, at the end of the arc, in the same PR as the fix. Hybrid, P1 baseline ->
stamped: overall P 0.117 -> 0.103, F1 0.208 -> 0.185, recall/MRR/NDCG 1.000 ->
0.974; keyword category P 0.135 -> 0.106, R 1.000 -> 0.938, F1 0.236 -> 0.189.
The stamped run also carries a corpus change (48 -> 49 documents, 44 -> 45
queries), so the per-fix attribution is in the progression table in
Tests/RAG_Eval/README.md rather than in this single diff. Gate re-run clean
afterwards: PASSED, 60 metrics within 0.05, 0 warnings.

Precision falls because the fix works: collapsing a document's chunks into one
fused row frees top-k slots that fill with further documents, and precision@k
divides by min(k, len(retrieved)). One passage per document is the intended
unit for an evidence list; the lost precision was partly counting duplicates.

AC #2 was amended, not waived - see the AC text. The short version: the
"rescue" half is blocked by a second, independent defect in the same fusion
function (the alpha blend buries FTS-only rows), now filed as TASK-4110 with a
measured before-number and a committed fixture that will show it closing.

**Post-review correction (final review of the fusion cluster).** AC #2 was
originally left as one TICKED box whose own text said half of it was unmet -
prose that told the truth attached to a checkbox that did not, which defeats
the DoD gate ("all `- [ ]` changed to `- [x]`") this task reached Done under: a
reader scanning boxes, or any tooling counting them, would have read the rescue
half as delivered. It is now split into #2a (merge half, ticked, 22-of-44
evidence) and #2b (rescue half, UNTICKED, delegated to TASK-4110), with the
explanatory prose preserved on both halves. The task remains Done: its DoD is
satisfied by the split being honest about scope - every criterion this branch's
commits actually deliver is ticked, and the one they do not is visibly open
against a filed task with a measured before-number - rather than by a box that
certifies work no commit performed.

**AC #2b CLOSED (2026-08-10, branch `fix/rag-fusion-weighting`, TASK-4110).**
The criterion, quoted: *"the FTS leg surfacing a relevant document the vector
leg ranks outside the top-k"*. It is now met, by a measured change to the
fusion weighting rather than to any wiring:

- **The fixture that was authored to falsify it now passes.**
  `kw-plant-maintenance-record` -> `note-saltmarsh-hide` in the eval harness:
  plain rank 1, semantic **absent** (still - pinned by
  `test_the_vector_blind_fixture_is_still_vector_blind`, so the corpus can
  still tell a rescue from ordinary vector coverage), engine FTS leg rank 1,
  hybrid **rank 8** - and rescued `fts-only` (`fts_rank=1`,
  `vector_rank=None`), i.e. by the blend, not by a merge with a vector twin
  and not by widening the candidate pool.
- **The structural claim, not just the one query.** The shipped `rrf_k = 5`
  makes an FTS-only rank-1 row strictly outrank vector-only rows from
  **rank 10** ((1-0.7)/(5+1) = 0.0500 > 0.7/(5+10) = 0.04667), which falls
  inside the ~20-row window `_hybrid_search` actually fuses. At the old
  `rrf_k = 60` the same boundary was vector rank ~83 - outside any window
  fusion ever sees, which is exactly why this half could not be delivered by
  the id-space fix.
- **Harness numbers, re-stamped in the same PR:** hybrid keyword recall
  0.938 -> 1.000, keyword NDCG 0.938 -> 0.957, overall recall 0.974 -> 1.000;
  no gated cell fell, and `semantic`/`plain` moved +0.000 on all 40 of their
  gated metrics.
- **Seen in the product**, on a real 64-document library through the running
  TUI: a document matched only by literal words took the last slot of a full
  Hybrid Basic result list, banded `keyword match`; with the constant
  reverted that slot held a vector row and the document was absent.
<!-- SECTION:NOTES:END -->
