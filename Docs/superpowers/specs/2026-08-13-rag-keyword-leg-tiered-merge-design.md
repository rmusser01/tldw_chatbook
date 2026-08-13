# RAG keyword leg — form-tiered sub-leg merge + the sweep re-run (TASK-15700)

Date: 2026-08-13
Status: draft-pending-user-review
Programme: RAG server-port (P0 #1428; P1 #1458; fusion #1469; weighting
#1487; P2ab #1517; 15400 #1574 — dev `31e56a5a1`)
Arc: the merge fix TASK-15400 filed as its real finding, plus the
re-application of that arc's pre-registered decision rule under the fixed
merge. TASK-15700's task file carries the full measured evidence and the
seven ACs — it is authority alongside this spec; read it first.

## The defect, one paragraph (measured, TASK-15400 Task 3)

`RAGService._keyword_search` merges the four sub-legs with
`interleave_rankings` — a round-robin over sub-leg POSITION in fixed
source order (media first). Fusion consumes the resulting LEG rank. So a
sub-leg's row count and the source ordering — properties with no relation
to the query — decide other sub-legs' leg ranks. Under widening
constructions this displaced an untouched AND rank-1 row to leg rank 2
behind another sub-leg's zero-confidence fallback row, killing the
vector-blind fixture's rescue and dropping scoped recall 1.000→0.429
(decomposed exactly: 4 note-targets behind a media fallback row; 3
media-targets safe because media interleaves first). This is what
disqualified both widening candidates in the 15400 sweep.

## The fix: two tiers, attributable, minimal

**Key structural fact that makes this small:** the fallback fires ONLY
when a sub-leg's primary form returns zero rows — so within one query,
each sub-leg's rows are ALL primary or ALL fallback. Tiering is therefore
a partition of sub-legs, not of rows.

- `_keyword_search` partitions the sub-leg result lists into **tier 1 =
  sub-legs whose rows came from the construction's PRIMARY form** and
  **tier 2 = sub-legs that fell back** (the seam knows which execution
  produced the rows — `_fts_rows_with_fallback` returns that fact today
  via the form stamp; surface it as a per-sub-leg flag rather than
  re-deriving from metadata). Round-robin WITHIN each tier exactly as
  today (rank-fair; raw FTS5 scores stay incomparable across sources);
  tier 1 wholly precedes tier 2.
- Fallback-ness is f(construction, form) — the 15400 Task-1 ruling: under
  `or`, or-form rows ARE the primary and tier 1; under `and_then_or`
  they are tier 2. The rule is WRITTEN DOWN at the merge site (AC#1's
  wording), with the incident (the displaced vector-blind fixture, the
  3/7 scoped decomposition) cited.
- **Byte-identity at the shipped default:** `and_stopword_trim` has no
  fallback — every sub-leg is tier 1 — so the merged order is IDENTICAL
  to today's for the shipped construction, the legacy `and`, and `or`
  (all-primary in all three). Pin it. Only `and_then_or` (and any future
  fallback-bearing construction) changes behavior.
- The AC#2 pin fails on the current round-robin: one sub-leg returns a
  primary rank-1 row, another returns many FALLBACK rows, and the
  primary row must lead the merged output. (An all-primary many-vs-one
  case must NOT change — the rank-fair semantics between primaries is
  correct and kept; pin that too so the fix cannot overreach.)
- `pipeline_builder_simple.py:370`'s twin call: plan-phase verification
  item — establish whether its fts_lists can ever carry mixed forms (if
  its legs never run fallback constructions, it needs only a comment
  naming why it is exempt; do not refactor it speculatively —
  TASK-3501 owns that unification).

## The re-run: the 15400 rule, verbatim, under the fixed merge

The 15400 sweep's four construction rows PLUS a fifth pre-registered row:
**`prefix`** (per-token `"tok"*`; AC#5 promotes or rejects its 3-rescue
lead on hybrid-level evidence — it widens as PRIMARY rows, so tiering
does not protect against it and the matrix must show what it displaces).

The decision rule is 15400's, unchanged and re-applied mechanically:
- Hard constraints: (a) `kw-plant-maintenance-record` keeps its hybrid
  rescue; (b) no gated cell regresses > 0.02 in any mode; (c) the
  escaping suite green (prefix quoting: star OUTSIDE the quotes, already
  proven in the 15400 probes; extend the suite to the prefix form).
- Winner = max census subject to constraints; tie-breaks as before
  (fewest extra FTS queries → smallest code delta); the
  negative-composition record rides along (still corpus-vacuous — say
  so, never "quiet").
- **If a widening construction now qualifies, it SHIPS as the new
  default** — that is the payoff this merge fix exists to unlock
  (`and_then_or`'s census was 29/53 with zero-row 11/60; scoped should
  hold 1.000 under tiering because the four note-target AND rows stay
  tier 1 — but the TABLE decides, not this sentence). If every widening
  construction still fails, the mechanism is recorded per AC#3 and
  `and_stopword_trim` stays — a null re-run is a recorded outcome.
- Staleness discriminator, census-vs-fusion caution, and the
  SHIPPED_CONTROL_CENSUS=20 control-row semantics all carry over from
  the 15400 machinery unchanged; the control row still runs `and`.

## AC#4 — the zero-headroom question, answered honestly

The counterfactual said a merge fix alone restores the vector-blind
rescue at slot 10 of 10. This arc's answer to "what provides margin":
NOTHING IN THIS ARC — fusion params (rrf_k/alpha/pool) own the margin
and are out of scope (the weighting arc's territory; retuning them
requires a new discriminating fixture per the P2ab headroom table). The
write-up states, from the winner's fused table: the fixture's slot, its
score gap to the first excluded row, and the named next displacer (one
more vector row, or any primary widening). If the winner's matrix shows
the fixture at the last slot, that fragility is stated in the README
headroom table beside the scoped 5/7-at-rank-9 note it already carries.

## Error handling

- The tier partition inherits every sub-leg degrade path unchanged (an
  erroring sub-leg is absent from both tiers, exactly as today).
- A construction with no fallback concept never produces a tier-2 entry
  (structural; pin).

## Out of scope (declared)

- Fusion-parameter changes (rrf_k/alpha/pool stay).
- pipeline_builder's twin merge beyond the exemption comment (3501).
- The sync cache twins (TASK-15701, filed).
- Score-aware cross-source merging (raw FTS5 scores are incomparable
  across sources — the tier + rank-fair design deliberately preserves
  that reasoning; a bm25-normalized merge is a different, unproposed
  arc).
- New fixtures; PRF and all later P2c candidates.

## Testing

- Always-on: the AC#2 displacement pin (fails on current round-robin);
  the all-primary byte-identity pin (shipped default order unchanged);
  the rank-fair-between-primaries pin; the no-tier-2-without-fallback
  structural pin; prefix-form escaping cases; mutations (tiering
  removed → AC#2 pin reds; tier order inverted → byte-identity pin
  reds).
- Gated: the five-row sweep + rule application; the re-measured
  residual census per category (AC#7); ONE deliberate re-stamp (AC#6)
  in the environment matching the committed fingerprint (the 15400
  method: main venv + PYTHONPATH to the worktree + import provenance
  asserted in-run), reconciled cell-by-cell against the winner's
  predicted movers.
- Live check per lessons-live-verification.md: the query class the
  winner unlocks, through Library RAG Answer on the hybrid profile,
  scratch profile + teardown; the 15400 live-check hole (RAG Answer
  4-min first query) is a known hazard — budget for it and report
  honestly if it recurs.
- Worktree venv discipline: assert `tldw_chatbook.__file__` resolves in
  the worktree before ANY measurement (the path-hook trap).

## Plan-phase verification items

1. `_fts_rows_with_fallback`'s exact return shape — surface the
   used-fallback flag without re-deriving from row metadata (and confirm
   the all-or-nothing-per-sub-leg fact holds for every sub-leg,
   including the two-statement conversations helper).
2. `interleave_rankings`' other caller (pipeline_builder:370-371) —
   mixed-form reachability; exemption comment or finding.
3. Whether tier-aware merging lives in `_keyword_search` (a partition +
   two interleave calls — no fusion.py change) or as a new fusion.py
   helper; prefer the former (smallest delta, the rule stays at the
   site AC#1 names).
4. The prefix construction's seam plumbing (a fifth valid value for
   `fts_match_construction`? or sweep-only?): if it can win, it must be
   shippable — so it enters the vocabulary + cache key + escaping suite
   like the others, gated behind the same validation.
5. The sweep machinery's row definitions (CONSTRUCTION_STRATEGIES) —
   adding the prefix row without disturbing the four existing rows'
   meaning; SHIPPED_CONTROL_CENSUS stays 20.
