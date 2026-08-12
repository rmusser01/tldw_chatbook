---
id: TASK-15400
title: Engine keyword-leg MATCH construction starves natural-language queries
status: In Progress
assignee: []
created_date: '2026-08-11'
updated_date: '2026-08-12 00:07'
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
- [ ] #1 The engine keyword leg's MATCH construction is decided on measured
      evidence, with the alternatives (AND-of-all-tokens, stopword-trimmed
      AND, OR-with-rank, `build_fts_match_query` reuse, and any hybrid of
      them) compared on the RAG_Eval golden set in all three modes — the
      table in this description is the starting point, not the answer
- [ ] #2 The chosen construction is applied at one seam shared by every FTS
      sub-leg (media, notes, conversations, prompts), not per sub-leg
- [ ] #3 The number of golden queries for which the keyword leg returns
      zero rows is reported before and after, per category, alongside the
      number for which it returns the TARGET (the two are far apart: OR
      rescues 34/40 to non-empty but only 10/40 to a hit)
- [ ] #4 `kw-plant-maintenance-record` keeps its plain and hybrid cells, or
      the fixture is deliberately re-authored with the reason recorded in
      both golden.toml and corpus.toml
- [ ] #5 `Tests/RAG_Search/test_fts5_query_escaping.py` stays green — each
      token remains individually quoted
- [ ] #6 Any movement in the committed baselines is a deliberate, disclosed
      re-stamp naming this task, produced by a full gated matrix re-run
- [ ] #7 The `prompt` category's cells are re-read afterwards: whether the
      shipped prompt queries become answerable is the decision point for
      whether those fixtures also need re-authoring
- [ ] #8 The two keyword paths (this engine leg vs the Library's four-seam
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
