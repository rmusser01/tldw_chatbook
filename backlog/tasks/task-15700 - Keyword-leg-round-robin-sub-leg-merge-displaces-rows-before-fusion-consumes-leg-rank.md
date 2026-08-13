---
id: TASK-15700
title: >-
  Keyword leg round-robin sub-leg merge displaces rows before fusion consumes
  leg rank
status: Done
assignee:
  - '@claude'
created_date: '2026-08-12 22:58'
updated_date: '2026-08-13 20:02'
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
- [x] #1 The keyword leg's cross-sub-leg merge no longer lets one sub-leg's row count decide another sub-leg's leg rank: the merged order is derived from a stated, attributable property of the rows (e.g. per-sub-leg rank with AND rows ahead of fallback rows, or a real score), and the rule is written down where the merge happens
- [x] #2 The behaviour is pinned by a test that fails on the CURRENT round-robin: a query where one sub-leg returns a rank-1 row and another returns many rows must keep the rank-1 row ahead of them in the merged output
- [x] #3 TASK-15400's four-construction sweep is re-run under the new merge and the table is published: for each construction, whether `kw-plant-maintenance-record` keeps its hybrid rescue and what scoped recall does. A widening construction that still fails is recorded with its mechanism rather than dropped
- [x] #4 The zero-headroom problem is addressed explicitly: if the vector-blind fixture is rescued only at the last fused slot, the write-up states what margin (if any) the change provides and what would displace it next
- [x] #5 The `prefix` construction is measured at BOTH leg and hybrid level under the new merge, so its 3-rescue lead is either promoted on evidence or rejected on evidence
- [x] #6 Any movement in `Tests/RAG_Eval`'s committed baselines is one deliberate, disclosed re-stamp naming this task, with before/after numbers in the PR and the environment-fingerprint decision stated (see the harness README's re-stamp sections)
- [x] #7 The residual "keyword leg returns zero rows for 39 of 60 golden queries" figure is re-measured and reported after the change, per category, alongside the leg-level census (21 of 53 today)
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
## WHAT SHIPPED (2026-08-13)

**Part A — the merge fix (AC#1, AC#2).** `RAGService._keyword_search`'s
cross-sub-leg gather is now FORM-TIERED: rows whose stamped `fts_match` form
is the construction's PRIMARY form (tier 1) wholly precede rows carrying its
FALLBACK form (tier 2), interleaved within each tier, truncated to top_k. The
rule is written at the gather site; the form is derived from one table,
`FTS_MATCH_FORMS_BY_CONSTRUCTION` (construction -> (primary, fallback)), never
hardcoded per branch, and an UNSTAMPED row fails safe into tier 1 under every
construction. AC#2's pin (`Tests/RAG_Search/test_keyword_leg_tiered_merge.py`)
was confirmed RED on the old round-robin with the incident's own shape.

**Part B — the default MATCH construction (AC#3, AC#5).**
`and_stopword_trim` -> **`and_then_prefix`** (full AND primary; per-token
prefix matching as the fallback for a sub-leg whose primary returned zero
rows).

## THE DECISION RECORD — the rule's winner and the ruling's, never conflated

1. **The pre-registered TASK-15400 rule was applied VERBATIM and ran to
   completion.** Census-maximal `and_then_or` (29) DISQUALIFIED on constraint
   (b); `or` on (a) and (b). Qualifiers {and_stopword_trim 21, prefix 23,
   and_then_prefix 23}; **max census 23 is a TIE**, and the two tied rows are
   measurement-identical on every captured axis (same census hit-set, 0/105
   gated cells moved, ALL 60 per-query hybrid top-10s and ALL 60 keyword-leg
   top-10s identical, `lost` 0 both ways). Tie-break 1 (fewest extra FTS
   STATEMENTS, measured 240 vs 460 over the 60 golden queries) **selected
   `prefix`.**
2. **The OWNER RULED `and_then_prefix` ships instead** (AskUserQuestion,
   2026-08-13), applying the standing stability-over-quick-wins ruling to a
   dimension the tie-break PREDATES: `prefix` widens as the PRIMARY, so Part
   A protects it not at all and it carries a MEASURED self-displacement shape
   (12 prefix-competitor docs + 1 exact-match doc, top_k=5 -> the exact doc
   absent, inside ONE sub-leg where tiering has nothing to tier);
   `and_then_prefix` is immune by construction (a non-empty AND primary is
   never widened).

**`and_then_prefix` is NOT the rule's own output and is never presented as
such** — disclosed at every site that records the outcome (`config.py`,
`_escape_fts5_query`, three test docstrings, the README), and the
shipped-default pin is named `test_the_shipped_default_is_the_owner_ruled_construction`
for exactly that reason.

**THE PRICE:** 220 extra SQLite statements over the 60-query set (460 vs
240) — **92% of sub-legs actually fall back**, an UPPER BOUND belonging to
the 172-document eval corpus and not a forecast (the fallback fires only
where the AND primary found nothing, so a denser corpus hits it less). Wall
time indistinguishable (~1.0-1.5s/row). ZERO measured retrieval difference
between the two qualifiers.

**NOT A SUPERSET BY CONSTRUCTION:** the new default's PRIMARY is the FULL AND
(function words included), so a sub-leg whose full AND returns rows never
seeks the trim-only hits. `lost` 0 is a MEASURED corpus fact, never a
structural guarantee.

## THE SIX-ROW TABLE (AC#3) — gated, hybrid, k=10, 60 golden queries

```
row              construction  census  resc  lost  zero  rescue  rank   gate (105 cells, 3 modes)
and                       and      20     0     0    40     yes     9   REGRESSED (5 past 0.02)
and_trim    and_stopword_trim      21     1     0    39     yes     9   PASSED  (0 moved)
or                         or      28     9     1    11      NO     -   REGRESSED (12 past 0.02)
and_or            and_then_or      29     9     0    11     yes    10   REGRESSED (8 past 0.02, 5 past 0.05)
prefix                 prefix      23     3     0    36     yes     9   PASSED  (0 moved)
and_pfx       and_then_prefix      23     3     0    36     yes     9   PASSED  (0 moved)
```

Constraint (b) was measured in ALL THREE modes by the gate's own
`compare_or_update` against the committed baselines; `plain` and `semantic`
moved zero cells under every construction. Scoped recall: 1.000 everywhere
except `or` (0.429).

**PART A'S PAYOFF, MEASURED:** `and_then_or` moved rescue **NO -> yes @ slot
10** and scoped **0.429 -> 1.000** — exactly the zero-headroom rescue this
task's description forecast, to the slot. Its leg form stamps show the
mechanism (`['and','or','or',...]`: the untouched AND primary leading its
nine tier-2 fallbacks).

**AND IT STILL DOES NOT SHIP,** now on (b) and by FUSION rather than by the
merge: tier 2 confines fallback rows inside the keyword LEG, but tier 2 still
enters fusion, where a fallback row carrying a vector rank becomes a MERGED
row outscoring any fts-only row (paraphrase/vocab mrr+ndcg, overall.mrr
-0.056, recall +0.000 — rank-1 answers re-ranked, not lost). Sharper: all
five regressing queries have EMPTY keyword legs under the shipped default, so
their leg is **100% tier 2**, tier 1 is empty, the partition is the identity
function — **Part A is structurally INERT on exactly the disqualifying
queries.** The residual 8-census gap is a fusion-weighting question, out of
this arc's scope and recorded with its mechanism so it need not be
re-derived.

**`or`'s failure was NEVER the merge:** the fixture is absent from the leg's
top-10 entirely, displaced inside the notes sub-leg's own bm25-ordered,
LIMITED result set before any merge is consulted. A widening PRIMARY puts
every row in tier 1, so tiering is a structural no-op there.

## AC#4 — the boundary, stated without headroom language

Under the shipped construction the vector-blind fixture holds **slot 9 of
10** — the same slot the outgoing default gave it. This is **not** margin.
The row immediately below is a **MATHEMATICAL TIE**: `(1-alpha)/(rrf_k+1) =
0.3/6` and `alpha/(rrf_k+9) = 0.7/14` are `== 1/20` **exactly** in rational
arithmetic (4 ULPs apart in IEEE-754, measured diff 6.938893903907228e-18).
The fixture keeps its slot on `reciprocal_rank_fusion`'s documented
`(-score, fts_rank, vector_rank)` tie-break, **not by margin**, and it sat on
that same boundary under the OLD default too. The gap to the first EXCLUDED
row (slot 11) is 0.003333333333333341 and **must never be printed as
headroom**. Next displacer = a MERGED (fts+vector) row, which is what a
widening PRIMARY manufactures (`and_then_or` inserts one at slot 9 and pushes
the fixture to slot 10). Nothing in this arc adds margin; fusion parameters
own it and stayed out of scope.

## AC#6 — the re-stamp is a DISCLOSED NON-EVENT

**Zero of 105 gated cells moved**, and stronger than a count: the gate's own
105 printed cell lines are **byte-identical** between a run at the old
default and a run at the new one. No baseline file was touched in this arc,
no stamp was manufactured, and the committed environment fingerprint is
TASK-15400's. The fixture files were deliberately NOT edited either
(`corpus_sha256` hashes their bytes, so even a comment would force a re-stamp
this arc did not earn) — which is why `golden.toml`'s dated 15400 receipts
still name `and_stopword_trim`.

## AC#7 — residual zero-row, RE-MEASURED per category at the shipped default

Leg-level, negatives included, 2026-08-13, `and_then_prefix`:

```
keyword 1/16 · negation 2/3 · negative 7/7 · paraphrase 13/13
prompt 4/5 · scoped 0/7 · vocabulary_mismatch 9/9      TOTAL 36 of 60 (was 39)
```

Leg-level census **23 of 53** (was 21; 20 pre-arc), per category: keyword
15/16 · negation 0/3 · paraphrase 0/13 · prompt 1/5 · scoped 7/7 ·
vocabulary_mismatch 0/9. The three ids gained against the pre-arc control are
named, not counted: `pm-vendor-chaser` (also reachable by the outgoing trim),
plus `kw-quillon-mast` ("guy tension" vs a document saying "tensions") and
`kw-thimble-relay` ("relay board swap" vs "swapping"/"swapped") — the prefix
fallback's own two. The shipped class in one line: a typed content word now
matches a document word that STARTS with it; the reverse does not hold.

## LIVE CHECK (2026-08-13) — what was demonstrated live, and what is bounded

Scratch profile (own `[paths] data_dir`, `HOME`/`XDG_*` redirected, model
cache read-only, `HF_HUB_OFFLINE=1`). Isolation confirmed at the running PID
(`lsof`: 0 handles under the real profile, 64 under the scratch) and the real
`config.toml` byte-identical before and after (sha256 `ea1f6cfb…`). Library
built through the app's own paths: 36 of this repo's User Guide pages written
with `add_note`, indexed with `index_entries`; the app listed "Notes (36)".

**Through the UI:** Library ▸ Search / RAG reached "mode: Search ⇄ ✓ RAG
Answer" and started a run — which **never returned**: "searching · Notes…" at
~98% CPU for **eleven minutes**, no Evidence row. That is the 2026-08-12
recurrence (4+ minutes then), now filed as **TASK-15810**; a CPU sample puts
the spin in the Python interpreter, NOT in model loading, so the old
"embedding stack came up" attribution is not established.

**Through the engine, against the same app-written notes and vector index,
two arms one config value apart** (the TASK-15400 precedent, and stated as
the fallback it is):

- `how do I schedule a watchlist brief` — shipped: primary
  `'"how" "do" "I" "schedule" "a" "watchlist" "brief"'` returns nothing, the
  fallback `'"schedule"* "watchlist"* "brief"*'` returns **1 row**
  (`watchlists.md`, stamped `prefix`); old default: `'"schedule" "watchlist"
  "brief"'`, **0 rows** — the page says "briefing", never "brief". Fused
  score 0.1167 → **0.1667** (fts-only + vector = merged row); rank unchanged
  at 1 because the vector leg already had it.
- `what does the anyone brief do` (vector leg misses the target) — shipped:
  `watchlists.md` at **rank 7 of 13**; old default: **absent**, 12 rows.
- Control `how do I change the color theme and appearance` — **identical 15
  rows, order and scores**, both arms.

**Bound:** on 36 real documentation pages the vector leg answers nearly
everything, so the gain reads as a keyword row plus a score rather than a
reordering — which is what 0-of-105 predicted. `Docs/User_Guide/library/
search-and-rag.md` carries the same three results and the same bound, stamped
at `db73f0953`.

Full tables, the rule line by line, and the reports:
`.superpowers/sdd/2026-08-13-rag-keyword-leg-tiered-merge/task-{1,2,3,4,5}-report.md` (session-local artifacts, not in the repo; the substantive tables are duplicated in this file and in Tests/RAG_Eval/README.md).
Harness record: `Tests/RAG_Eval/README.md` ("The fifth arc, and the re-stamp
that did NOT happen").

---

### Task-3's own sweep note, kept as written — read it WITH the ruling above

Its "WINNER under the rule = `prefix`" is the RULE's output, which is exactly
what the owner overrode; `and_then_prefix` is what shipped.

Task-3 six-row sweep (gated, 2026-08-13, tiered merge): census and 20 / and_trim 21 / or 28 / and_or 29 / prefix 23 / and_pfx 23; control self-check 20/53; 105/105 (+0.000) at the shipped default before the run. PART A PAYOFF MEASURED: and_then_or moved rescue NO -> yes@slot 10 and scoped 0.429 -> 1.000 (leg form stamps ['and','or',...] show the AND primary leading its tier-2 fallbacks). `or` still fails and the merge was never its cause — the fixture is absent from the leg top-10 entirely (intra-sub-leg self-displacement; a widening PRIMARY is all tier 1, so tiering is a no-op). Hard constraint (b) measured in ALL THREE modes by the gate's own compare_or_update: prefix and and_then_prefix PASSED with 0 of 105 cells moved; and_then_or REGRESSED (8 cells past 0.02, 5 past the 0.05 fail band — paraphrase/vocab mrr+ndcg, overall.mrr -0.056), or REGRESSED 12. and_then_or's failure is a FUSION boundary, not a tiering shortfall: all five regressing queries have EMPTY keyword legs under the shipped default, so their leg is 100% tier 2 and Part A is structurally INERT on exactly the disqualifying queries. WINNER under the rule = `prefix` (census 23, +kw-quillon-mast +kw-thimble-relay, lost 0; tie with and_pfx at 23 — the two are identical on all 60 hybrid top-10s AND all 60 keyword-leg top-10s — broken by fewest-extra-FTS-STATEMENTS, measured 240 vs 460 over the 60 golden queries, i.e. 220 extra / 92% of sub-legs falling back on this SPARSE corpus, an upper bound not a forecast; wall time indistinguishable). The census-maximal and_then_or is disqualified on (b). AC#7: residual zero-row 36/60 (from 39), census 23/53 (from 21). AC#4: fixture at slot 9 of 10 — NOT headroom: it sits EXACTLY ON THE alpha/rrf_k BOUNDARY (its 0.3/6 == the vector-rank-9 row's 0.7/14 == 1/20 in exact arithmetic; 4 ULPs apart in float) and is held there by reciprocal_rank_fusion's documented (-score, fts_rank, vector_rank) tie-break, not by margin. The gap to the first EXCLUDED row (slot 11, conv-lower-store-flood 0.04666666666666666) is 0.003333333333333341 and must NOT be printed as headroom. Next displacer is a MERGED fts+vector row, not another vector-only row. Negative axis corpus-VACUOUS (7/7 negatives zero leg rows under every construction). Winner moves 0 of 105 cells, so AC#6's re-stamp is a disclosed non-event. Full tables + the rule line-by-line: .superpowers/sdd/2026-08-13-rag-keyword-leg-tiered-merge/task-3-report.md
<!-- SECTION:NOTES:END -->
