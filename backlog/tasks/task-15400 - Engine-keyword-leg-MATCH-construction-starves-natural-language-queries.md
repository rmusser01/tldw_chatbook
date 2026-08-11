---
id: task-15400
title: Engine keyword-leg MATCH construction starves natural-language queries
status: To Do
assignee: []
created_date: '2026-08-11'
labels:
  - rag
  - retrieval
dependencies: []
priority: high
---

## Description

The RAG engine's keyword (FTS5) leg builds its MATCH expression as an
implicit AND over **every** token of the user's query, function words
included (`RAGService._escape_fts5_query`, chosen in TASK-3995 over
whole-query phrase-quoting), and applies no plural/singular widening. A
document must therefore contain literally every word the user typed —
"about", "into", "that" — to be returned at all.

Measured over the RAG_Eval golden set during TASK-15020/B2 (2026-08-11):
the keyword leg returns **zero rows for 40 of the 60 golden queries**. It
fires only where the query happens to be keyword-shaped (`keyword` 13/16
targets found by the FTS leg alone, `scoped` 7/7) and never for the
natural-language classes (`paraphrase` 0/13, `vocabulary_mismatch` 0/9,
`negation` 0/3, `prompt` 0/5).

For media, notes and conversations this is invisible: the semantic leg
answers those queries and hybrid looks healthy. It became visible with B2's
prompts sub-leg, because prompts have no vector index at all — the FTS leg
is their only path — so the whole `prompt` category reads recall 0.000 in
every mode while the sub-leg demonstrably works (a keyword-shaped query
returns the right prompt at hybrid rank 9 on the same runtime).

The Library's own four-seam keyword path already solved half of this with
`library_fts_query.build_fts_match_query` (plural/singular widening), which
the engine leg does not use — so the two keyword paths in this app answer
the same query differently.

This is a retrieval-behaviour change affecting every hybrid query, so it
needs its own before/after measurement rather than a drive-by fix.

## Acceptance Criteria

- [ ] The engine keyword leg's MATCH construction is decided on measured
      evidence, with the alternatives (AND-of-all-tokens, stopword-trimmed
      AND, OR-with-rank, `build_fts_match_query` reuse) compared on the
      RAG_Eval golden set in all three modes
- [ ] The chosen construction is applied at one seam shared by every FTS
      sub-leg (media, notes, conversations, prompts), not per sub-leg
- [ ] The number of golden queries for which the keyword leg returns zero
      rows is reported before and after, per category
- [ ] Any movement in the committed baselines is a deliberate, disclosed
      re-stamp naming this task
- [ ] The `prompt` category's cells are re-read afterwards: whether the
      shipped prompt queries become answerable is the decision point for
      whether those fixtures also need re-authoring
- [ ] The two keyword paths (engine leg vs the Library's four-seam path)
      either share a construction or the divergence is documented with its
      reason
