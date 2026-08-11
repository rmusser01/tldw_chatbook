---
id: TASK-15020
title: 'RAG P2ab: instrument renewal + P0-deferred constraints'
status: Done
assignee: []
created_date: '2026-08-11 04:37'
updated_date: '2026-08-11 19:49'
labels:
  - rag
  - eval-harness
  - hybrid
  - p2
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After the weighting arc, hybrid recall/MRR/NDCG sit at 1.000 on every scored eval query, so the harness can only detect regression, not measure improvement. This arc restores that power with fail-first fixture authoring (a candidate is admitted only when today's pipeline measurably fails it) and, inside the same branch, lands the three P0-deferred constraints the renewed harness measures: scoped searches silently dropping to semantic-only, prompts having no keyword-leg coverage at all, and the Library canvas's window ignoring the active profile's default_top_k. Programme: RAG server-port P2, first of two P2 arcs (the second is P2c measured feature admission).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 New candidate fixture categories (compositional/multi-hop, negation-sensitive, acronym-without-context, precision-pressure) are admitted only where measured to fail today's pipeline (target misses top-10 in every vector-bearing mode); a class that cannot be authored to fail is recorded as unfailable in the README rather than force-fit.
- [x] #2 The scoped category ships with a routing before-pin: scoped golden queries are recorded as failing by routing (the P0 semantic-only-under-scope constraint) prior to B1, giving the scope-aware hybrid fix a documented before-number.
- [x] #3 The eval corpus scales to ~150 documents while every existing fixture stays byte-identical, and a probe confirms each pre-existing golden query's top-10 is unchanged (or the new doc is reworded) after the corpus addition.
- [x] #4 Scope-aware hybrid removes the engine's metadata_allowlist raise for hybrid search, retiring the scoped-to-semantic-only disclosure family (ROUTE_NOTE_HYBRID_SCOPED and its route-note/User Guide copy) app-wide.
- [x] #5 A read-only prompts keyword sub-leg is added to the hybrid engine (Prompts DB FTS5), inventoried and vocabulary-pinned (source_type: prompt) per the chacha private-sqlite pattern, with prompt fixtures' before-state recorded as total absence across all modes.
- [x] #6 The Library canvas's default (unset/invalid) top_k resolves to the active profile's default_top_k instead of the fixed literal 5, while an explicit user-set top_k value keeps winning unchanged.
- [x] #7 One deliberate re-stamp closes the sub-arc, replacing the at-ceiling README warning with a per-category headroom table showing the new categories' honest (lower) baselines and the scoped category's post-B1 scores.
- [x] #8 A live TUI check confirms all three user-visible changes: a scoped hybrid search returns keyword-found in-scope evidence, a prompts hit surfaces in hybrid results, and the Library evidence list remains usable at the profile's default depth (15 rows).
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Spec: Docs/superpowers/specs/2026-08-11-rag-p2ab-instrument-and-deferred-constraints-design.md. Plan: Docs/superpowers/plans/2026-08-11-rag-p2ab-instrument-and-deferred-constraints.md. Sequencing per the plan's 9 tasks: Half A (harness scope machinery -> fail-first authoring + ~150-doc scale-up) lands first; then B1 (scope-aware hybrid; TASK-14752's coverage-copy fix rides inside B1's disclosure-seam work) -> B2 (prompts sub-leg) -> B3 (Library window honors profile); ONE deliberate re-stamp closes the sub-arc with a per-category headroom table.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Nine tasks on feat/rag-p2a-instrument-renewal (26b970d19..HEAD). Half A renewed the instrument; Half B landed the three P0-deferred constraints; one deliberate re-stamp closed it.

APPROACH. Half A: harness scope machinery (real EffectiveScope built from fixture slugs, passed to the seam's own scope= parameter), then FAIL-FIRST authoring — 31 candidates probed against the real stack, admitted only where the target missed the top-10 in BOTH vector-bearing modes, each carrying an "# admitted:" receipt. Corpus 49->172 (first 49 byte-identical), golden 45->60. Half B: B1 pushed metadata allowlists into the engine's FTS sub-legs (json_each id filters, fail-closed on unenforceable keys) and deleted the Library scope->semantic divert; B2 added a read-only prompts FTS sub-leg on the chacha private-sqlite pattern; B3 made the Library window's depth the active profile's default_top_k. Every intermediate gated run read environment_changed and was deliberately NOT re-stamped.

WHAT THE INSTRUMENT NOW MEASURES. Hybrid overall recall came off the 1.000 ceiling to 0.826 — the harness can measure improvement again, which was the arc's whole point. Scoped hybrid flipped 0.000 -> 1.000 (B1, measured). negation 0.000 and prompt 0.000 are the two cells with headroom; the README's new headroom table names them as P2c's admission targets.

EVIDENCE HIGHLIGHTS.
- The re-stamp's every delta reconciles to five named classes; nothing was stamped over as a surprise. The semantic precision drops are the P@k denominator filling, verified by measurement (mean_docs_at_k 9.105 -> 9.652) with recall/MRR/NDCG byte-identical; the hybrid keyword MRR/NDCG -0.001 is exactly one relevant rank moving 8->9 ((1/8-1/9)/16 = 0.00087).
- Two P2c feature premises RETIRED with measured negatives: acronym (MiniLM bridges MTBF/PPE/BOM/RTO/UPS unaided, ranks 1,1,1,2,2) and compositional (ranks 1,1,1,1,2,2) are UNFAILABLE on this corpus and model. Anchor dilution did not move either. All 16 rejections are preserved in golden.toml so the rulings are auditable without re-authoring.
- Scope SIZE is the scoped class's lever and was measured: 32 docs in scope -> 1 of 8 candidates failed, 80 -> 6, 100 -> 7. Review caught the guard floored at 40 and re-probed the shipped seven there: only 4 of 7 still failed, i.e. a trim would have raised the before-number 0.000 -> ~0.43 with every test green. Now pinned at exactly 100 and digest-pinned.
- B1's rescue mechanism was verified from the engine's own recorded provenance, not paper arithmetic: 5 of 7 are FTS-only and reach rank 9; 2 of 7 sit at vector rank 12 and 20 inside the over-fetched pool and lead at rrf_k=60. The counterfactual at the old constant is 0.286, not 0.000.
- B2's sub-leg is proven reachable end-to-end on the same runtime (prompt at hybrid rank 9, fts_rank 1 / vector_rank None read off metadata).
- Live TUI check passed all three arms with verified teardown (config sha256 identical before/after, zero real-profile handles, real DB mtimes untouched).

TECHNICAL DECISIONS AND TRADE-OFFS.
- Scoped queries stay OUT of the overall row, but the rationale was REPLACED: the old routing reason died with B1 (the modes are comparable now); the surviving reason is the denominator — a scoped query is asked over 100 documents, every other over 172.
- B3 shipped resolve_active_rag_top_k() (depth-only, torch-free) instead of mirroring the Console seam, because mirroring would have put `import torch` on a UI render (0.98s measured). Pinned by a subprocess assertion that torch is not imported.
- One deliberate divergence: the window clamps a >50 profile to 50, Console stays uncapped. Pinned as a PAIR after review showed a mutation capping both arms left 199 tests green.
- precision_pressure was not authored: the class needs many near-relevant decoys AND a crisp label boundary, and those pull against each other.

HONEST BOUNDS, STATED.
- prompt 0.000 in all three modes is a bound on the engine's MATCH construction (TASK-15400, High, filed with measured attribution), NOT a B2 failure. The keyword leg returns zero rows for 40 of 60 golden queries; prompts have no vector leg to mask it. Fixtures were deliberately NOT reworded after seeing the result.
- negation 0.000 is a genuine open capability gap — P2c's target.
- The scoped flip is real but FRAGILE: 5 of 7 targets land at rank 9, one rank from k=10's edge, and their 9-vs-10 placement turns on a 1-ULP float win. Inclusion has a real 0.0033 margin; placement does not.
- AC#3 DEVIATION, adjudicated ACCEPT: full top-10 identity for the old queries was measured, not achieved, and is not achievable when a corpus triples. What held is every old query's relevant ranks (one moved 8->9, still rescued) and every negative's behaviour. Nothing was reworded because nothing needed it.
- AC#6 PREMISE CORRECTED: the canvas has no top_k control at all, so "an explicit user value keeps winning" lives at the API level and is pinned there with a call-counter.
- Floor-inert gated cells went 10/60 -> 45/105 — the deliberate cost of fail-first authoring, recorded in the README.

TESTS. Ungated Tests/RAG_Eval 225 passed / 10 skipped; gated 235 passed with the gate reading PASSED: No regression on 105 metrics. Full battery green apart from three PRE-EXISTING failures characterised and filed (TASK-13214 re-confirmed with a second sighting, TASK-15500, TASK-15501).

MODIFIED/ADDED (principal): Tests/RAG_Eval/{fixtures/corpus.toml,fixtures/golden.toml,baselines/*.json,README.md,harness/*.py} and its test modules; tldw_chatbook/RAG_Search/simplified/ engine + fusion seam; Library/library_local_rag_search_service.py, library_rag_state.py, active_config.py; UI/Screens/chat_screen.py delegation; Docs/User_Guide/library/search-and-rag.md.
<!-- SECTION:NOTES:END -->
