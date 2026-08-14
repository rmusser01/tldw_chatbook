# RAG plain path — rank-fair cross-seam merge (TASK-16071)

Date: 2026-08-14
Status: draft-pending-user-review
Programme: RAG server-port (eight arcs merged; last: the PRF null #1622 —
dev `8f9e7302a`)
Arc: the transferable finding the PRF null filed, fixed. TASK-16071's task
file carries the measured evidence, the code citations, and three worked
examples — it is authority alongside this spec; read it first.

## The defect (measured, code-cited — the 16071 filing)

The Library's plain four-seam keyword path gathers per-seam results and
merges them by CONCATENATION in fixed source order
(`library_local_rag_search_service.py:450-452` — `rows.extend` over
`_KNOWN_KEYWORD_SOURCE_TYPES = ("notes","media","conversations",
"prompts")`), with per-seam `limit=top_k` and `"score": None` on all FOUR
row builders (incl. the buried `_prompt_row`). There is NO cross-seam
ranking: every notes row precedes every media row regardless of match
quality, so any downstream cut (the harness's doc-level k, RAG Answer's
evidence cap, a fused window) buries non-note targets under seam-fill.
Measured: under widened passes the oracle observability was 8/22 (TF) /
15/22 (rare-term) with conversation targets at merged positions 19-21;
under the SHIPPED pass the burial is bounded by per-seam volume but the
privilege is structural. The semantic arm of the same service SORTS and
truncates (`:677-678`) — the keyword arm is the only unranked merge left.

## The fix: the engine's own proven primitive, imported

Replace the ordered concatenation with **rank-fair round-robin**
(`interleave_rankings` from `RAG_Search/fusion.py` — the exact primitive
the engine keyword leg has used since the fusion cluster, keyed on the
same doc identity the engine uses): position 0 of each seam, then
position 1, in seam order within a position. Raw FTS5 scores across
different tables stay incomparable (the engine's documented reasoning —
a bm25 cross-seam SCORE merge is explicitly rejected by precedent), so
rank position is the only meaningful cross-source signal, exactly as the
engine concluded for its own sub-legs.

- NO tiering machinery: the four-seam path builds its MATCH via
  `build_fts_match_query` with no fallback constructions — every seam is
  "primary" — so this is the pre-15700 all-primary interleave, which
  15700 proved correct for that regime. If the four-seam path ever
  gains fallback forms, the 15700 tier design applies; a comment at the
  merge site says so.
- The rule is WRITTEN DOWN at the merge site with the incident cited
  (the 16071 worked examples: kw-quillon-mast's media target at merged
  14 behind 13 notes; the conv targets at 19-21).
- Dedup across seams: structurally vacuous (seams disjoint by
  source_type — same argument as 15700's, one comment, no machinery),
  but `interleave_rankings` requires a key — verified at review: rows
  carry `source_id` + `provenance.source_type` (`_note_row` et al.,
  ~L1080-1136), so the key is `(provenance.source_type, source_id)`.
- Callers verified at review: `_search_keyword` is consumed only inside
  the service (`:268` direct plain, `:331` the profile arm); the
  route-note/count consumers read the outcome dict, not raw row order —
  the enumeration item narrows to display-side consumption only.
- The merged list still carries NO cross-seam truncation at this site
  (today's contract: consumers cut). The change is ORDER only.

## What the instrument can and cannot see (pre-registered)

- Plain cells CAN move: reordering changes which rows sit inside any
  downstream cut. Expected movers, stated before the run: plain
  keyword/scoped MRR/NDCG may shift where multi-seam results interleave;
  plain keyword RECALL may RISE if a currently-buried non-note target
  crosses into the harness's k (the two 0.875-misses' mechanism is a
  PLAN-PHASE VERIFICATION item — establish whether they are seam-burial
  or genuine misses BEFORE predicting). Paraphrase/vocab plain cells
  CANNOT move (0.000 because zero rows — an ordering change cannot
  create rows). Semantic/hybrid cells MUST NOT move (the engine's
  hybrid keyword leg does not route through this path — the twin
  exemption verified in 15700; pre-register hybrid/semantic at +0.000
  as the zero-movement proof).
- THE CONTROL RE-RUN (not a re-verdict): the PRF probe's oracle
  observability table re-measured under the new merge — expected: the
  seam-burial component vanishes (conversation targets no longer parked
  at 19-21 by order alone); whatever remains is per-seam volume. PRF
  STAYS RETIRED — the re-run recalibrates the four-seam bound for
  future candidates (16072, HyDE), it does not reopen the null.
- ONE deliberate re-stamp if cells move (expected); the fingerprint-
  matching environment method; reconciled cell-by-cell against the
  pre-registered movers; an unpredicted mover is a STOP.

## User-visible surface (scope it tight)

The Library Search evidence list displays grouped per source ("top 15
per source") — PLAN-PHASE VERIFICATION: how the display consumes row
order (task-4023-era work shaped it). If the UI groups by source_type
independently of list order, the retrieval reorder is invisible in
Search mode and visible only where a cross-seam cut exists (RAG Answer
evidence, fused windows) — the honest user-facing claim is scoped to
those. The User Guide sentence updates only if its claims change; stamp
after the live check.

## Out of scope (declared)

- Cross-seam SCORE merging (bm25 incomparability — rejected by the
  engine's own precedent, recorded in the filing).
- The engine keyword leg (already rank-fair + tiered).
- Any widening construction change; PRF re-admission (retired).
- TASK-16072 (clarification gate) and the agentic-document-expansion
  roadmap task (filed separately; see Task 1).
- Chat-RAG's pipeline_builder twin (its own interleave is already
  rank-fair; verified in 15700's exemption).

## Testing

- Always-on: the AC-style displacement pin (a rank-1 media row must
  precede rank-5 notes rows in the merged output — RED on today's
  concatenation); all-seams-equal rank-fairness pin; single-seam
  byte-identity (a query matching only notes produces an unchanged
  list); the no-truncation contract pin; mutations (interleave reverted
  to extend → the displacement pin reds; key dropped → dedup pin if
  applicable).
- Gated: the pre-registered movers + zero-movement proof; the control
  re-run table; the re-stamp.
- Live check per lessons-live-verification.md: a query whose best match
  is a media/conversation doc surfacing above notes seam-fill in RAG
  Answer evidence (scratch profile; TASK-15810 hazard budgeted; the
  engine A/B fallback sanctioned and stamped if the UI hangs).
- Environment: fresh worktree — the PINNED venv recipe; import
  provenance asserted before every measurement; the re-stamp in the
  main venv.

## Plan-phase verification items

1. The two plain-keyword misses (0.875 = 14/16): seam-burial or genuine?
   (Decides the recall prediction.)
2. The evidence-list display's consumption of row order (per-source
   grouping vs list order).
3. `interleave_rankings`' key: the service-side doc identity (what the
   four-seam rows carry; the engine uses `_fusion_doc_key` — the
   service needs its equivalent or a thin adapter).
4. Whether ANY other consumer of `_search_keyword`'s row order exists
   (grep the callers; the route-note/count headline code reads the list
   — order-sensitive display strings must be enumerated as disclosed
   changes if they shift).
5. The PRF-probe control re-run's exact invocation (the machinery is on
   dev; the oracle table must run under the new merge without
   re-opening the verdict path — the PRE-REGISTERED-selector guard
   already enforces this).
