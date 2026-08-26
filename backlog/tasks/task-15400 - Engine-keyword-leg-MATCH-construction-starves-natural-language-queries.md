---
id: TASK-15400
title: Engine keyword-leg MATCH construction starves natural-language queries
status: Done
assignee: []
created_date: '2026-08-11'
updated_date: '2026-08-12 23:23'
labels:
  - rag
  - retrieval
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The RAG engine's keyword (FTS5) leg builds its MATCH expression as an
implicit **AND over every query token** (`RAGService._escape_fts5_query`,
chosen in TASK-3995 over whole-query phrase-quoting). A document must
therefore contain literally every word the user typed to be returned at all.

Measured over the RAG_Eval golden set during TASK-15020/B2 (2026-08-11):
the keyword leg returns **zero rows for 40 of the 60 golden queries**. It
fires only where the query happens to be keyword-shaped (`keyword` 13/16
targets found by the FTS leg alone, `scoped` 7/7) and never for the
natural-language classes (`paraphrase` 0/13, `vocabulary_mismatch` 0/9,
`negation` 0/3, `prompt` 0/5).

**The dominant cause is AND-strictness over CONTENT words, not function
words.** That distinction was measured, because it decides which fix is
worth building:

| MATCH construction | rescues (of the 40 zero-row queries) | returns the TARGET |
|---|---|---|
| stopword-trimmed AND | **1/40** (`pm-vendor-chaser`) | 1/40 |
| `build_fts_match_query` (the Library's four-seam construction) | **1/40** (`kw-quillon-mast`) | 1/40 |
| OR-of-tokens | **34/40** | 10/40 |

Trimming function words is a ~2.5% lever: `pm-vendor-chaser` is the only
query in the set blocked *solely* by one ("about"). The rest miss on
content words — `template`, `building`, `rough`, `turns`, `pulls`, `builds`
— which no stopword list removes. Function words are a contributing factor,
not the mechanism.

A second, smaller factor is the absence of plural/singular widening: the
engine leg has none, so `terms` does not match a document's `term`
(`pm-glossary`).

For media, notes and conversations this whole thing is invisible: the
semantic leg answers those queries and hybrid looks healthy. It became
visible with TASK-15020/B2's prompts sub-leg, because prompts have no
vector index at all — the FTS leg is their only path — so the entire
`prompt` category reads recall 0.000 in every mode while the sub-leg
demonstrably works (a keyword-shaped query returns the right prompt at
hybrid rank 9 on the same runtime).

This is a retrieval-behaviour change affecting every hybrid query, so it
needs its own before/after measurement rather than a drive-by fix.

## Constraints (measured — read before choosing a construction)

Any widening has to survive both of these, and the highest-recall option
already fails the first:

1. **PRECISION — the vector-blind fixture.** `kw-plant-maintenance-record`
   ("plant maintenance record" -> `note-saltmarsh-hide`) is the golden
   set's only query whose hybrid cell is a real before-number rather than a
   repeat of semantic's: plain finds it at rank 1, semantic never returns
   it. Its design rests on AND-of-terms uniqueness — the phrase reads as
   plant-and-equipment upkeep (the shared subject of ~20 documents) while
   the target's "plant" is botanical. **Measured: OR-of-tokens LOSES it**
   (target rank 1 -> not returned within k=10). golden.toml's own note
   warns to read the corpus section of the same name before editing either
   side; a rare-identifier version of this case was tried first and failed.
   Net effect of OR across the whole set: targets found by the leg go
   20/60 -> 29/60, i.e. **+10 gained, 1 lost** — so OR is a real
   improvement, not a free one, and the loss is exactly the fixture that
   exists to detect this.
2. **INJECTION-SAFETY — the per-token quoting is load-bearing.** An
   unquoted token FTS5 parses as column-filter or operator syntax (the
   hyphenated-numeric `Obsidian-3` raises
   `OperationalError('no such column: 3')`). `Tests/RAG_Search/
   test_fts5_query_escaping.py` (10 tests) reds if that property breaks.
   Any new construction must keep quoting each token; only the JOIN between
   them is in play.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The engine keyword leg's MATCH construction is decided on measured
      evidence, with the alternatives (AND-of-all-tokens, stopword-trimmed
      AND, OR-with-rank, `build_fts_match_query` reuse, and any hybrid of
      them) compared on the RAG_Eval golden set in all three modes — the
      table in this description is the starting point, not the answer
- [x] #2 The chosen construction is applied at one seam shared by every FTS
      sub-leg (media, notes, conversations, prompts), not per sub-leg
- [x] #3 The number of golden queries for which the keyword leg returns
      zero rows is reported before and after, per category, alongside the
      number for which it returns the TARGET (the two are far apart: OR
      rescues 34/40 to non-empty but only 10/40 to a hit)
- [x] #4 `kw-plant-maintenance-record` keeps its plain and hybrid cells, or
      the fixture is deliberately re-authored with the reason recorded in
      both golden.toml and corpus.toml
- [x] #5 `Tests/RAG_Search/test_fts5_query_escaping.py` stays green — each
      token remains individually quoted
- [x] #6 Any movement in the committed baselines is a deliberate, disclosed
      re-stamp naming this task, produced by a full gated matrix re-run
- [x] #7 The `prompt` category's cells are re-read afterwards: whether the
      shipped prompt queries become answerable is the decision point for
      whether those fixtures also need re-authoring
- [x] #8 The two keyword paths (this engine leg vs the Library's four-seam
      path) either share a construction or the divergence is documented
      with its reason. **TASK-3997 is the four-seam half of the same
      question** (it found `build_fts_match_query`'s AND-join zeroing
      results on the P1 set) — the two should be decided together or one
      should explicitly defer to the other
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Arc plan: Docs/superpowers/plans/2026-08-11-rag-keyword-leg-match-construction.md (5 tasks). Spec (authority): Docs/superpowers/specs/2026-08-11-rag-keyword-leg-match-construction-design.md.

1. Construction seam (engine): SearchConfig.fts_match_construction (and | and_stopword_trim | or | and_then_or, default 'and', NOT TOML-wired), RAGService._fts5_match_expressions -> (primary, fallback|None), _FTS5_STOPWORDS, per-sub-leg zero-row fallback loop + metadata['fts_match'] provenance, construction in the hybrid/keyword cache key ('and' byte-identical to pre-arc keys).
2. Sweep construction axis: fusion_sweep Strategy gains the construction field, CONSTRUCTION_STRATEGIES (4 rows at shipped fusion params), leg-level census + negative-composition counters, control-row self-check (census must reproduce the shipped 20).
3. THE SWEEP RUN: gated four-row sweep + NEAR/prefix probes; apply the spec's pre-registered mechanical rule in writing; winner computed, never chosen. Null result ships the table.
4. Ship the winner: default flips to the sweep's winner; disclosed oracle updates; docstring's under-review block resolved from the TABLE.
5. Re-stamp + closure: one deliberate baseline re-stamp, README census/headroom prose, live TUI check (natural-language prompt query through Library RAG Answer), 15400 Done with the sweep table verbatim.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Task-3 sweep run (gated, 2026-08-11): census and 20 / and_trim 21 / or 28 / and_or 29; control self-check reproduced the shipped 20/53. Hard constraint (a) DISQUALIFIES both or AND and_then_or — the vector-blind fixture kw-plant-maintenance-record loses its hybrid rescue under both (mech absent), and (b) is violated too (scoped/recall 1.000 -> 0.429, keyword/recall -0.062, vocab+paraphrase MRR -0.13/-0.115). The spec's by-construction argument was true of the SUB-LEG and false of the LEG: _keyword_search merges sub-legs round-robin (interleave_rankings), so other sub-legs falling back demotes the untouched AND row from leg rank 1 to 2, and fusion consumes leg rank (0.05 -> 0.0429, below the vector row at rank 11). The scoped collapse decomposes exactly the same way: the 4 note-targeted scoped queries drop behind a media fallback row, the 3 media-targeted ones survive — 3/7 = 0.429, the measured cell. Computed WINNER under the pre-registered rule = and_stopword_trim (census 21, +pm-vendor-chaser; prompt gated pin flips 0.000 -> 0.200; zero regressions in any mode; 0 extra FTS queries). Escaping suite 37 green; plain/semantic cells byte-identical across all four constructions; 105 gated metrics all +0.000 (informational on this machine — environment mismatch).

READ THE WIN AS WHAT IT IS: the arc was raised on "the keyword leg returns zero rows for 40 of 60 golden queries" and the winner moves that to 39/60 — closer in substance to a null result plus a re-scoped merge-level finding than to a fix. The candidates that DO move the census (28-29, prompt 5/5) are exactly the ones the constraints reject, and they are rejected by the MERGE, not by the MATCH form. Re-fusing the and_or pass with the fixture restored to leg rank 1 and nothing else changed puts it back at slot 10 of 10 — a merge-ordering fix rescues it with ZERO headroom, so "fix interleave_rankings" is not sufficient on its own either. Full table: .superpowers/sdd/2026-08-11-rag-keyword-leg-match-construction/task-3-report.md

## Closing note (Task 5, 2026-08-12) — SHIPPED, re-stamped, closed

**The sweep table (gated, reproduced identically in three separate runs):**

```
row              construction  census  resc  zero     P@k     R@k     MRR    NDCG  rescue  rank
and                       and      20     0    40   0.087   0.826   0.807   0.811     yes     9
and_trim    and_stopword_trim      21     1    39   0.089   0.848   0.809   0.817     yes     9
or                         or      28     9    11   0.089   0.848   0.751   0.774      NO     -
and_or            and_then_or      29     9    11   0.089   0.848   0.751   0.774      NO     -
```

`SearchConfig.fts_match_construction` now defaults to **`and_stopword_trim`**
— the maximum census subject to the three pre-registered hard constraints.
Both higher-census rows are disqualified, for DIFFERENT reasons: under `or`
the JOIN loses the vector-blind fixture (leg rank 11 in the k=20 fusion
window); under `and_then_or` the fixture's sub-leg is untouched and the
round-robin MERGE loses it. Constraint (a) is a property of the merge, not of
the match form — filed as **TASK-15700**, with the counterfactual margin
(restoring leg rank 1 puts the fixture back at slot 10 OF 10, zero headroom,
so a merge fix is necessary and not sufficient).

**THE NULL-ADJACENCY READING, kept in front of the numbers.** This arc opened
on "the keyword leg returns zero rows for 40 of 60 golden queries". The
winner moves that to **39 of 60**. One query. The candidates that move the
census properly (28-29, prompt 5/5) are exactly the ones the constraints
reject. Read this as a measured null-adjacent result plus a re-scoped
merge-level finding, not as a fix to the arc's opening complaint.

**The re-stamp.** One deliberate re-stamp; exactly **10 of the 105 gated
metrics** moved, all UP, all hybrid (5 `category.prompt.*` + 5 `overall.*`);
the other 95 at +0.000; nothing regressed in any mode. Fresh gated run reads
`PASSED: No regression. 105 metric(s) within 0.05 of baseline.` **Environment
reconciliation (explicit, because a stamp bakes the fingerprint):** the stamp
was run in the interpreter whose fingerprint MATCHES the committed baselines
(transformers 5.6.2 / torch 2.11.0 / chromadb 1.5.8), with `PYTHONPATH` forced
to this worktree and import provenance asserted in-run — so the `environment`
block did not move and the gate stays live for everyone on the shipped stack.
The branch's own worktree venv (5.15.0 / 2.13.0 / 1.5.9) still reads
ENVIRONMENT_CHANGED and gates nothing; it reproduced all 105 cells at +0.000
against these baselines, which is the evidence that the version gap is
numerically inert here.

**AC#7 (the prompt cells, re-read).** No golden query was re-worded on either
side. Four of the five prompt queries still miss in all three modes on absent
CONTENT words. The fifth, `pm-vendor-chaser`, is now answered at hybrid rank 9
and is deliberately KEPT as a **retained positive** — the admission rule would
reject it as a new candidate today, and re-authoring it fail-first would
delete the arc's only measured evidence while returning the cell to 0.000. Its
dated `# admitted:` receipt is intact; a `# retained:` receipt beside it
records the conversion.

**AC#8 (the two keyword paths).** They diverge deliberately and the reason is
recorded in `Tests/RAG_Eval/README.md`'s known-defects list: the engine leg's
construction was chosen by a HYBRID-FUSION measurement (every constraint that
decided it is about rows competing inside a fused top-k), and the Library
four-seam path has no fusion, no ranking and no leg to be displaced in, so its
construction needs its own evidence — TASK-3997, which now carries the number
to start from (`build_fts_match_query` rescues 1 of the 40).

**THE USER-VISIBLE CHANGE, re-measured rather than asserted** (three shapes,
only one of which moved):

* **only function words** ("what about the") — UNCHANGED: trimming everything
  empties the token list and the construction falls back to the full AND.
* **only content words** ("wombat burrow spindle") — BYTE-IDENTICAL: nothing
  to trim.
* **MIXED** ("notes about the vendor" -> `'"notes" "vendor"'`) — the ONLY case
  that widened, and the whole mechanism of the flip.

So: a query containing function words now matches more documents than before;
a query without them is untouched. Extra keyword rows can displace fused rows
(the same mechanism TASK-15700 owns) — it moved no gated cell down on the
golden set, and **no large or non-golden library was ever measured**. The
construction joins the cache key, so the flip costs one cold miss per query
and leaves pre-flip entries keyed apart rather than served to the new
construction.

**Live check (2026-08-12, scratch profile, isolation verified by `lsof` and a
byte-identical live config before/after).** Three prompts written through the
app's own `add_prompt`; two RAG profiles differing in exactly one value.
Library ▸ Search / RAG listed "Prompts (3)" and RAG Answer mode started a run
— which did NOT complete (the first hybrid query in a fresh profile sat on
"searching · Prompts…" at 98% CPU for 4+ minutes bringing the embedding stack
up), so no Evidence row was seen on screen. What IS demonstrated, on the same
app-written FTS index through the engine's own MATCH builder under those two
profiles: `"saved" "prompt" "for" "chasing" "a" "supplier" "about" "a" "late"
"order"` returns **0 rows**; `"saved" "prompt" "chasing" "supplier" "late"
"order"` returns **1** — the right prompt. Also learned at the surface: the
screen's plain **Search** mode is the four-seam path and is unaffected by this
change. `Docs/User_Guide/library/search-and-rag.md` is updated and stamped
with exactly this scope.

**Follow-ups filed:** TASK-15700 (the merge — the arc's real finding, carrying
the measured displacement, the exact scoped decomposition, the zero-headroom
counterfactual, and the unexplored `prefix` lead with its displacement risk
stated); TASK-15701 (`SimpleRAGCache`'s sync `get`/`put` render a key omitting
three search-defining dimensions — latent, no production caller, but the flip
turned it from a missed-hit risk into a mislabelled-entry risk).
<!-- SECTION:NOTES:END -->
