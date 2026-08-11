---
id: TASK-4110
title: >-
  Hybrid fusion alpha blend buries FTS-only rows below the vector leg, so hybrid
  never rescues a keyword-only match
status: Done
assignee: []
created_date: '2026-08-09 20:22'
updated_date: '2026-08-11 02:47'
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
- [x] #1 A document that only the keyword leg finds can appear in hybrid's top-k when the vector leg returns k or more distinct documents (the case that is structurally impossible today - a thin vector leg already lets keyword-only rows through and is not evidence)
- [x] #2 The chosen weighting is justified against a measurement, not asserted: the before/after numbers for the affected golden queries appear in the PR
- [x] #3 Tests/RAG_Eval baselines are re-stamped in the same PR, and the hybrid cell for kw-plant-maintenance-record moves from miss to hit
- [x] #4 A regression test pins that an FTS-only row outranks at least one vector-only row under the shipped default alpha and rrf_k
- [x] #5 A test asserts the vector-blind fixture is STILL vector-blind - semantic mode does not return note-saltmarsh-hide for kw-plant-maintenance-record - so a future model bump or re-stamp cannot silently return the corpus to the state where it cannot distinguish coverage from noise
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-08-10-rag-fusion-weighting.md (Task 4 sweep + Task 5 ship-the-winner) and Docs/superpowers/specs/2026-08-10-rag-fusion-weighting-design.md (decision rule + two rescue senses) for the measured fix.
<!-- SECTION:PLAN:END -->

## Implementation Notes (Task 5 — shipping the winner)

<!-- SECTION:NOTES:BEGIN -->
**Shipped `rrf_k = 5`** at one authoritative site: `SearchConfig.rrf_k` now defaults to the new `DEFAULT_HYBRID_RRF_K` in `RAG_Search/simplified/config.py`. `hybrid_pool_multiplier` (2) and `hybrid_alpha` (0.7) are untouched. AC #3 (baseline re-stamp) is the closing task's; everything else is done.

**The measurement (Task 4's sweep, `RAG_EVAL=1 pytest Tests/RAG_Eval/test_fusion_sweep.py -s`).** Six strategies over 45 golden queries (38 scored), hybrid mode: keyword recall@10 **0.938 → 1.000**, keyword NDCG **0.938 → 0.957**, overall recall **0.974 → 1.000**, and the vector-blind fixture (`kw-plant-maintenance-record` → `note-saltmarsh-hide`) goes from **absent to rank 8**, rescued `fts-only` — by the weighting, not by the pool. No per-category cell regressed (worst gated delta +0.000). Re-run under the shipped default, the harness gate reports PASSED with every hybrid cell up and semantic/plain byte-identical.

**Why 5 and not the server's 60.** The server calibrates k for candidate pools of thousands; `_hybrid_search` fuses `top_k * hybrid_pool_multiplier` ≈ 20 rows per leg. At k=60 an FTS-only rank-1 row scores 0.3/61 and only outranks vector-only rows from rank ~83 — outside the window fusion ever sees, which is the structural starvation this task describes. At k=5 it strictly outranks vector-only rows from rank **10** (0.0500 > 0.04667) — the honest boundary to quote. Rank 9 is the exact equality point (`3/10 × 1/6 == 7/10 × 1/14 == 1/20`); the keyword row still ranks above it, but in the floats the code computes it wins by one ULP on score (`(1.0-0.7)*(1/6)` rounds to 0.05, `0.7*(1/14)` is 0.049999999999999996), so the sort never reaches the documented `(-score, fts_rank, vector_rank)` tie-break there. Three separate pins: the weighting (rank 10), the boundary (rank 9, with the arithmetic it actually depends on), and the tie-break convention on a bit-identical pair.

**Scale bound, stated rather than buried.** "Nothing regressed" is bounded to a 49-document corpus where every scored query except the fixture was already at rank 1 — there was almost nothing left to damage. A 12× reduction in k re-weights the whole tail: vector ranks 9-20 stop being reachable by an FTS-only row's rank alone. On this corpus those slots hold irrelevant rows; at production scale they may not. The *structural* guarantee does survive scale (the ~20-row window is corpus-size-independent); the reassurance does not. Task 6's live check on a real library owns that question.

**Considered and declined: `hybrid_pool_multiplier = 3`.** `k5+pool3` beat `k5` by +0.005 on keyword MRR/NDCG — which, re-counted, is the *already-rescued* fixture moving rank 8 → 5, not a second document found (recall is already 1.000 under `k5` and does not move). The cost is a permanent +50% retrieval width on every hybrid query, compounding on the semantic leg via `SEARCH_RESULT_MULTIPLIER`. Declined: it has not earned it, which is also what the decision rule concluded mechanically.

**Both live fusion paths moved together.** Task 3 gave `resolve_rrf_k` an active-profile fallback, so `pipeline_builder_simple`'s legacy TASK-3501 blend now fuses at 5 as well, though it was never measured. Deliberate (two paths must not disagree on one measured number), disclosed on TASK-3501's file, and pinned by a new test.

**One resolver, one fallback.** Every fallback in `resolve_rrf_k` — unreadable profile, profile with nothing to say, non-numeric value, negative value — is the shipped `DEFAULT_HYBRID_RRF_K` (5). All of those inputs are reachable from user configuration (a TOML pipeline's `steps[].config.rrf_k`, a round-tripped `config.search.rrf_k`), so any of them falling back to 60 would silently revert *that one path* to the weighting this task measured away from. `fusion.DEFAULT_RRF_K` (60) survives only as `reciprocal_rank_fusion`'s own no-config signature default and its negative-k sanitization — a pure-library invariant for a caller who supplied nothing, unreachable from the live call sites, which always resolve through `resolve_rrf_k` first.

**Who picks the new default up — everyone; no migration needed.** Builtin profiles (including the default `hybrid_basic`) are constructed in code from a bare `RAGConfig()` and inherit it. Saved *custom* profiles are JSON, rebuilt via `SearchConfig(**search_data)` — and `search.rrf_k` did not exist before this branch (`fd6e3e323`, never on `origin/dev`), so a pre-branch profile JSON has no `rrf_k` member and falls through to the new default too (probed: legacy JSON rebuilds at 5). The only artifact that can freeze 60 is a profile saved from an *intermediate build of this unreleased branch* — a dev-machine concern, not a user-facing one. No migration work is owed.

**Oracle updates, all disclosed:** `test_defaults_unchanged` → `test_shipped_defaults` (pinned `rrf_k == 60`, now pins 5 plus the unchanged fallback), and four `TestPipelineRrfMerge` cases in `Tests/RAG/test_fusion.py` that hand-computed `1/(60+rank)` on the profile-resolved path (now `PROFILE_K`). Nothing else moved: `local_citation_capture` fixtures record k per row and were unaffected; `Tests/RAG_Eval` baselines are NOT re-stamped here.

**Files:** `RAG_Search/simplified/config.py` (the default + its justification), `RAG_Search/fusion.py`, `RAG_Search/simplified/rag_service.py`, `RAG_Search/pipeline_builder_simple.py`, `Library/library_rag_score_kinds.py`, `Library/library_rag_state.py`, `UI/Views/RAGSearch/search_handoff.py`, `Event_Handlers/Chat_Events/chat_rag_events.py` (docstrings that asserted k=60 or the ~0.016 fused ceiling), `Docs/Development/RAG/RAG-Documentation.md`, `backlog/decisions/005-*` (addendum), new `Tests/RAG_Search/test_fusion_rescue_pin.py`, plus `Tests/RAG_Eval/test_harness_run.py` (AC#5 gated guard).

## Implementation Notes (Task 6 — re-stamp, live check, closure)

**AC #3, the deliberate re-stamp.** One re-stamp at the end of the arc, in the
same commit as the docs and the closures, with both sets of numbers recorded.
Hybrid moved on 10 of its 20 gated cells and every move is a rise: keyword
recall **0.938 -> 1.000**, keyword NDCG 0.938 -> 0.957, keyword MRR 0.938 ->
0.945, keyword P 0.106 -> 0.113, keyword F1 0.189 -> 0.201; overall recall
0.974 -> 1.000, NDCG 0.974 -> 0.982, MRR 0.974 -> 0.977, P 0.103 -> 0.105, F1
0.185 -> 0.190. **`semantic` and `plain` moved +0.000 on all 40 of their gated
metrics** - the check that the change touched hybrid fusion and nothing else,
and the reason a moved semantic cell would have stopped the stamp. The headline
cell: `kw-plant-maintenance-record`'s hybrid result goes **miss -> hit at rank
8**, rescued `fts-only`, probed directly at the shipped default rather than
inferred from the recall aggregate. The gate re-run afterwards against the new
baselines reports `PASSED: No regression. 60 metric(s) within 0.05 of
baseline.` Two report-only movers, both explained rather than absorbed: hybrid
negatives' `max_top_score` 0.0115 -> 0.1167 is the fused score's *scale*
(`1/(rrf_k+1)`) moving, not confidence (`max_top_vector_score` is unchanged at
0.2387), and the latency block churns on process order, as the harness README
already documents.

**The live check answered the question the corpus could not.** The eval corpus
is 49 documents at k=10 with every other query already at rank 1; the product's
Library Search/RAG surface runs at `LIBRARY_RAG_DEFAULT_TOP_K = 5`, i.e. half
the fused candidate window. Run against the real Library DBs the check would
have been vacuous - that library has **four** indexed documents (450 of its 453
embeddings are chunks of one), so the vector leg cannot fill a result list and
keyword-only rows already pass, which this task's own description says is not
evidence. So 60 real documentation files were ingested into the scratch profile
through the app's own indexing path first. On the resulting 64-document
library, through the running TUI, default Hybrid Basic, default top-5: the
query "drafted expert" (two words occurring literally in one document, never
returned by vector retrieval) placed that document at position **5, banded
`keyword match`**, with positions 1-4 unchanged; with the constant reverted to
60 and nothing else changed, that slot held an ordinary vector row and the
document was absent. Ordinary semantic queries were unaffected - one query
returned a byte-identical list under both constants. The weighting moves the
bottom of the list, not the ranking above it. Measured over 274 vector-missed
queries on that library, the k change rescued 9 of them into the visible top-5,
always into the last slot, which is what the arithmetic predicts.

**What the numbers still do not cover.** "Nothing regressed" remains bounded:
one corpus, one embedding model, and now one real 64-document library. And the
eval corpus is now at its ceiling - with the rescue query answered, hybrid
recall is 1.000 on every scored query, so a future fusion retune can only be
measured here for *regression*. Any later weighting change that claims to add
something needs a **new** vector-blind fixture, authored the way this one was.

**Files (Task 6):** `Tests/RAG_Eval/baselines/{hybrid,plain,semantic}.json`
(re-stamped), `Tests/RAG_Eval/README.md` (progression table + the TASK-4110
entry retired + the shared-k note + the ceiling and scale bounds),
`Docs/User_Guide/library/search-and-rag.md` (the fused-score ceiling, the
keyword-only-row behaviour, and a live-check stamp),
`Tests/RAG_Search/test_hybrid_fusion_metadata.py` (a module docstring still
quoting the old ceiling), `backlog/tasks/task-3994 ...` (AC #2b closed with
evidence).
<!-- SECTION:NOTES:END -->
