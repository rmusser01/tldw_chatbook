---
id: TASK-17755
title: Adopt and_then_prefix on the four-seam keyword path
status: Done
assignee: []
created_date: '2026-08-18'
labels: [rag, retrieval]
dependencies: []
priority: medium
---

## Description (the why)

TASK-3997 investigated the four-seam (plain Search) path's AND-strictness and
the owner took the decision on 2026-08-18: **adopt `and_then_prefix`** — the
construction the engine leg has shipped since TASK-15700. That task was scoped
to investigate and propose; this one implements.

The measured case (`Docs/superpowers/qa/2026-08-18-four-seam-and-strictness/report.md`,
172-doc corpus, 53 ground-truthed golden queries):

| construction | MRR | zero-row queries |
|---|---|---|
| AND-strict (shipped) | 0.396 | 32 of 53 |
| pure prefix OR | 0.261 | 0 |
| **`and_then_prefix`** | **0.423** | 25 |

Two properties make this the low-risk arm. The 21 queries whose AND primary
already returns a row are **untouched by construction** — the fallback only
fires where the primary returned nothing — so there is no regression path to
the answers AND currently gets right. And pure OR was measured *worse* than
the status quo, which kills the naive alternative rather than leaving it as a
plausible-sounding option.

It also ends a live divergence: the Library screen's **Search** mode is this
path while **RAG Answer** is the engine leg, so one screen has two matching
rules today (an inflection miss answers in one and returns nothing in the
other).

## Acceptance Criteria (the what)

- [x] The four-seam keyword path applies an `and_then_prefix` construction:
      the existing AND-of-variant-groups stays the primary, and a per-token
      prefix form runs **only** for a sub-leg whose primary returned zero rows
- [x] A query whose AND primary returns rows produces byte-identical results
      to today — pinned by a test, since this is the property that makes the
      change low-risk
- [x] The zero-row rescue is demonstrated on the golden set, and the measured
      MRR does not fall below the AND-strict baseline (0.396 on the corpus in
      TASK-3997's report)
- [x] Plain Search and RAG Answer answer the same inflection-miss query on the
      same corpus — the divergence TASK-3997 documented is gone
- [x] The gated retrieval suite reads `PASSED: No regression. 105 metric(s)`;
      note that its `plain` cells CAN legitimately move here, unlike in
      reranking arcs — if they do, the move is the deliverable and the
      baselines are re-stamped deliberately with the reason recorded

## Implementation Plan (the how)

1. Extract the engine's PREFIX form and function-word list into one shared,
   dependency-free module rather than copying either.
2. Give `library_fts_query` a `build_prefix_match_query` adapter over it.
3. Write the zero-row fallback ONCE and route all four seams through it.
4. RED-first tests: never-built on a hitting sub-leg, rescue on an empty one,
   per-sub-leg independence, merge contract, cross-path agreement.
5. Run the gate; re-stamp only the baselines that actually moved.

## Implementation Notes

**What shipped.** The Library's four-seam plain-Search path now runs
`and_then_prefix`: `build_fts_match_query`'s AND-of-variant-groups stays the
primary, and a per-token PREFIX form (`"tok"*`, function words trimmed) runs
only for a sub-leg whose own primary returned zero rows. The decision is per
sub-leg, matching `RAGService._fts_rows_with_fallback` — one search can carry
AND rows from notes and prefix rows from media, which is the point.

**Reuse, not duplication.** The prefix builder and the 67-word function-word
list were *moved* to a new pure module, `Utils/fts5_match_forms.py`, and both
consumers import it: `rag_service._FTS5_STOPWORDS` is now that same object
(the name is kept because the engine's tests and the RAG_Eval probes read it),
and its inline prefix construction is now a call to the shared builder. It
lives under `Utils/` because that package's `__init__` is empty, so the pure
`library_fts_query` module stays cheap to import — `RAG_Search/__init__` pulls
the whole simplified engine. `Tests/RAG_Search/test_fts5_match_forms_shared.py`
asserts the stopword list is one OBJECT, not two that agree, so a
re-introduced copy reds while the copies still match.

**The fallback is written once** (`_rows_with_prefix_fallback`), taking a
"run this MATCH and give me rows" callable; the four seams supply only their
own service call. Each seam's `try/except` now wraps the whole helper, so the
error contract (`(True, [])` plus one log) is unchanged and no new failure
mode is added.

**Low-risk property, pinned by construction not by output.** The prefix form
is built lazily, after the primary comes back empty, so a hitting sub-leg
never constructs it — the tests count constructions and DB calls rather than
comparing rows, because a row comparison cannot tell "the fallback correctly
did not fire" from "it fired and returned the same thing".

**Deliberately NOT tiered.** TASK-16071's merge comment said the TASK-15700
tier design would apply here if this path ever gained fallback forms. It now
has them, and the merge is still untiered on purpose: `and_then_prefix` was
*measured* untiered (TASK-3997), and tiering would ship an unmeasured
ordering. The comment at the merge site now records that as a live decision
with a re-measure requirement, and the new suite pins the untiered order so
it cannot change silently.

**Measured effect (gated harness, 172 docs / 60 golden queries).** Only
`plain` cells moved, all up; every `semantic` and `hybrid` cell held at
+0.000:

| cell | before | after |
|---|---|---|
| plain overall mrr / precision | 0.304 | 0.326 |
| plain overall ndcg | 0.296 | 0.318 |
| plain overall f1 | 0.297 | 0.319 |
| plain overall recall | 0.293 | 0.315 |
| plain category.keyword.* | 0.844–0.875 | 0.906–0.938 |
| plain mean_docs_at_k | 0.304 | 0.457 |

Census: zero-row golden queries 39 → 37, one-row 21 → 22, ≥2 rows 0 → 1.
`plain.json` re-stamped deliberately; `semantic.json`/`hybrid.json` restored
untouched because no metric in them moved.

**One collateral premise had to be re-derived.** `test_cross_encoder_probe_run`
asserted `plain_census.reorderable == 0` and `row_order_changes == 0` — both
resting on the structural fact that AND-strictness never returned two rows, so
a reranker could not permute anything. One rescued query now returns several
rows and the cross-encoder duly permuted it (5 positions, 1 query). The arc's
actual conclusion survives intact: **every plain metric still moves by exactly
+0.000, MRR and NDCG included.** The census pin is re-stamped to the new
measured `== 1` (still an exact equality) and the row-order pin now asserts
`queries_reordered <= reorderable`, which keeps the part that is still
structural — a reranker cannot reorder a window it was never given.

**Modified/added:** `tldw_chatbook/Utils/fts5_match_forms.py` (new),
`Library/library_fts_query.py`, `Library/library_local_rag_search_service.py`,
`RAG_Search/simplified/rag_service.py`,
`Tests/Library/test_library_keyword_and_then_prefix.py` (new),
`Tests/RAG_Search/test_fts5_match_forms_shared.py` (new),
`Tests/RAG_Eval/test_cross_encoder_probe_run.py`,
`Tests/RAG_Eval/baselines/plain.json`, `Tests/RAG_Eval/README.md`.

## Final-review corrections (2026-08-18, applied before merge)

- **F1 — the rescue is 1 query, not 7.** TASK-3997's "7 of 32" came from an
  analytically-composed arm (whole-query fallback, a crude prefix form) that
  was not the shipped construction (per-sub-leg fallback, the engine's
  stopword-trimmed prefix), and its harness was never committed. The
  delivered rescue is `kw-thimble-relay` with the right document — which is
  the ENTIRE +0.022 MRR / +0.062 keyword-category move — plus
  `ng-mains-supply` gaining 6 non-relevant rows. TASK-3997's record is
  corrected in place. **The decision stands on the gate's own measurement**,
  which the reviewer reproduced including a control mutation that restores
  the old baseline bit-for-bit.
- **F2 — the "≥2 rows: 0 → 1" census line** is that negation query gaining
  non-relevant rows, invisible to the gate's scored cells. Disclosed here
  rather than left to look like a second rescue. Mitigating: the engine leg
  has returned those same rows since TASK-15700, so this is the convergence
  this arc exists to produce.
- **F3 — the re-stamp carries a latency move** the report did not mention:
  plain mean 14.2 → 18.4 ms, p95 24.8 → 31.1 (reproduced independently at
  18.6 / 30.8). Expected — a zero-row sub-leg now runs a second query — and
  recorded so the next person reading the baseline diff does not have to
  rediscover it.
- **F5 — AC#3's instrument.** The 0.396 figure from TASK-3997 was never
  re-measured on the shipped code; AC#3 is satisfied by the GATE's plain
  cells (0.304 → 0.326, all 105 metrics re-verified) rather than by that
  number. TASK-17955's AC#1 was also vacuous as filed — tiering is provably
  the identity on this corpus — and now requires the fixture to be extended
  first.
- **F4** (`rag_service._FTS5_STOPWORDS` is now an inert monkeypatch point) is
  left as-is: it is the shared object, and the identity guard is what keeps
  it honest.
