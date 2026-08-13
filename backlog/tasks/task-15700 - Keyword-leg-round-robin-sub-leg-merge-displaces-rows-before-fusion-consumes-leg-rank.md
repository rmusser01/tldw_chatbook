---
id: TASK-15700
title: >-
  Keyword leg round-robin sub-leg merge displaces rows before fusion consumes
  leg rank
status: In Progress
assignee:
  - '@claude'
created_date: '2026-08-12 22:58'
updated_date: '2026-08-13 16:27'
labels:
  - rag
  - retrieval
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The RAG engine's keyword leg (`RAGService._keyword_search`) runs four
source sub-legs — media, notes, conversations, prompts — and merges them
with `interleave_rankings`: a **round-robin over sub-leg position**, not a
merge over any score, rank quality or relevance signal. Hybrid fusion then
consumes the resulting **leg** rank.

The consequence is that *how many rows one sub-leg returns changes the leg
rank of every other sub-leg's rows*. A document its own sub-leg ranked
first can be pushed out of the fused top-k by rows from a sub-leg that has
nothing to do with the query, and the ordering is decided by the fixed
source order (media first) rather than by anything about the documents.

This was found by measurement, not by reading: it is what disqualified both
widening candidates in TASK-15400's MATCH-construction sweep, and it is why
that arc could only ship a one-query improvement. **The blocker on the
keyword leg is the merge, not the match form.** TASK-15400's own spec
argued its widest candidate was safe "by construction: a nonempty AND never
falls back" — that premise was verified TRUE at the sub-leg and the
conclusion was still FALSE, because the guarantee is about a SUB-LEG and
the constraint is about the LEG.

### The measured evidence (TASK-15400 Task 3, gated run over the 172-doc golden corpus)

**The vector-blind fixture.** For `kw-plant-maintenance-record` →
`note-saltmarsh-hide` under the `and_then_or` construction, the notes
sub-leg's row is *untouched* — still stamped `and`, still the sub-leg's
rank 1. The media and conversations sub-legs return zero AND rows for that
query, fall back to OR, and inject 10 rows each; the round-robin puts media
FIRST, so the untouched notes row is demoted from leg rank 1 to leg rank 2.
Fusion (`alpha` 0.7, `rrf_k` 5) then reads:

```
fts-only, leg rank 1   : 0.05                    ((1-0.7) * 1/6)
vector-only, vec rank 9: 0.049999999999999996    (0.7 * 1/14)   <- strictly loses, by 6.94e-18
fts-only, leg rank 2   : 0.042857142857142864
vector-only, vec rank 11: 0.04375                                <- now strictly wins
```

One position of cross-sub-leg displacement is the whole distance between
"rescued at slot 9" and "absent from the top-10".

**The scoped category, which decomposes exactly.** Measured under each
scoped query's real `EffectiveScope`, the same displacement moves scoped
recall 1.000 → 0.429: the four NOTE-targeted scoped queries each drop to
leg rank 2 behind a media fallback row, while the three MEDIA-targeted ones
keep leg rank 1 — because media is first in the round-robin. **3 of 7 =
0.429, the measured cell to the digit.** The collapse is not statistical;
it is the interleave's source-type order.

### The counterfactual margin — a merge fix alone is NOT sufficient

Re-fusing the same `and_then_or` pass with the fixture restored to leg rank
1 and **nothing else changed** puts it back — at **slot 10 of 10**. At leg
rank 1 it scores 0.05, the media row it displaces falls to 0.0428…, and the
next contender is the vector rank-9 row at 0.049999999999999996. So a
merge-ordering fix rescues this fixture with **zero headroom**: the next
widening step would displace it again. Any plan here must treat "fix
`interleave_rankings`" as necessary and not sufficient, and must say what
provides the margin.

### The unexplored lead, with its risk stated

A **prefix**-matching construction (`"tok"*` per token) was probed
report-only over the same 40 zero-row queries and rescues **3**
(`kw-quillon-mast`, `kw-thimble-relay`, `pm-vendor-chaser`) against the
shipped `and_stopword_trim`'s 1, with **0 negatives gaining rows**. It is
the only unexplored variant that beats the shipped construction. Two
caveats are load-bearing: the probe is **leg-level only** (its hybrid cells
were never measured), and it widens rows exactly the way the OR forms do —
so it carries **the same displacement risk described above, unmeasured**.
It is a candidate for the sweep this task re-runs, not a shortcut around it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The keyword leg's cross-sub-leg merge no longer lets one sub-leg's row count decide another sub-leg's leg rank: the merged order is derived from a stated, attributable property of the rows (e.g. per-sub-leg rank with AND rows ahead of fallback rows, or a real score), and the rule is written down where the merge happens
- [ ] #2 The behaviour is pinned by a test that fails on the CURRENT round-robin: a query where one sub-leg returns a rank-1 row and another returns many rows must keep the rank-1 row ahead of them in the merged output
- [ ] #3 TASK-15400's four-construction sweep is re-run under the new merge and the table is published: for each construction, whether `kw-plant-maintenance-record` keeps its hybrid rescue and what scoped recall does. A widening construction that still fails is recorded with its mechanism rather than dropped
- [ ] #4 The zero-headroom problem is addressed explicitly: if the vector-blind fixture is rescued only at the last fused slot, the write-up states what margin (if any) the change provides and what would displace it next
- [ ] #5 The `prefix` construction is measured at BOTH leg and hybrid level under the new merge, so its 3-rescue lead is either promoted on evidence or rejected on evidence
- [ ] #6 Any movement in `Tests/RAG_Eval`'s committed baselines is one deliberate, disclosed re-stamp naming this task, with before/after numbers in the PR and the environment-fingerprint decision stated (see the harness README's re-stamp sections)
- [ ] #7 The residual "keyword leg returns zero rows for 39 of 60 golden queries" figure is re-measured and reported after the change, per category, alongside the leg-level census (21 of 53 today)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Spec: Docs/superpowers/specs/2026-08-13-rag-keyword-leg-tiered-merge-design.md
Plan: Docs/superpowers/plans/2026-08-13-rag-keyword-leg-tiered-merge.md
Part A (Task 1): form-tiered sub-leg merge at _keyword_search's gather site (tier 1 = primary-form sub-legs, tier 2 = fallback sub-legs; interleave within each tier; tier 1 wholly precedes tier 2; truncate to top_k). Pre-registered intermediate gate: 105/105 at (+0.000) and control census 20.
Part B (Tasks 2-4): prefix + and_then_prefix constructions, six-row sweep under the 15400 decision rule verbatim, conditional default flip.
Task 5: conditional re-stamp, README/AC#4 margin, live check, closure.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Task-3 six-row sweep (gated, 2026-08-13, tiered merge): census and 20 / and_trim 21 / or 28 / and_or 29 / prefix 23 / and_pfx 23; control self-check 20/53; 105/105 (+0.000) at the shipped default before the run. PART A PAYOFF MEASURED: and_then_or moved rescue NO -> yes@slot 10 and scoped 0.429 -> 1.000 (leg form stamps ['and','or',...] show the AND primary leading its tier-2 fallbacks). `or` still fails and the merge was never its cause — the fixture is absent from the leg top-10 entirely (intra-sub-leg self-displacement; a widening PRIMARY is all tier 1, so tiering is a no-op). Hard constraint (b) measured in ALL THREE modes by the gate's own compare_or_update: prefix and and_then_prefix PASSED with 0 of 105 cells moved; and_then_or REGRESSED (8 cells past 0.02, 5 past the 0.05 fail band — paraphrase/vocab mrr+ndcg, overall.mrr -0.056), or REGRESSED 12. WINNER under the rule = `prefix` (census 23, +kw-quillon-mast +kw-thimble-relay, lost 0, 0 extra FTS statements; tie with and_pfx at 23 broken by fewest-extra-queries, measured 0/60 vs 60/60 fallback expressions). The census-maximal and_then_or is disqualified on (b). AC#7: residual zero-row 36/60 (from 39), census 23/53 (from 21). AC#4: fixture at slot 9 of 10, score 0.05, gap 0.003333333333333341 to the first excluded row (conv-lower-store-flood, 0.04666666666666666); next displacer is a MERGED fts+vector row, not another vector-only row. Negative axis corpus-VACUOUS (7/7 negatives zero leg rows under every construction). Winner moves 0 of 105 cells, so AC#6's re-stamp is a disclosed non-event. Full tables + the rule line-by-line: .superpowers/sdd/2026-08-13-rag-keyword-leg-tiered-merge/task-3-report.md
<!-- SECTION:NOTES:END -->
