# RAG P2c opener — the keyword-leg MATCH construction, measured (TASK-15400)

Date: 2026-08-11
Status: draft-pending-user-review
Programme: RAG server-port (P0 #1428; P1 #1458; fusion #1469; weighting
#1487; P2ab #1517 — dev `3b1ad8eff`)
Arc: the first P2c admission candidate. P2c's discipline is "a feature is
admitted only if it moves the renewed instrument"; this arc applies that
discipline to a MEASURED, FILED defect before any ported feature gets its
turn — because it bounds what every later P2c candidate can show.

## The problem (measured, not asserted — TASK-15400's census)

`_escape_fts5_query` (rag_service.py ~L3082) builds AND-of-every-token:
each token individually quoted (the load-bearing injection-safety from
TASK-3995), joined by implicit AND. Correct for rare-literal queries;
fatal for natural-language ones. Census on the 172-doc corpus: **the
engine keyword leg returns ZERO rows for 40 of 60 golden queries**
(keyword 13/16 hit, scoped 7/7, paraphrase 0/13, vocabulary_mismatch 0/9,
negation 0/3, prompt 0/5, negative 0/7-correctly). The semantic leg masks
this for every source type except prompts (keyword-only by design — B2's
rescue path works, but no natural-language query ever reaches it).

Measured attribution (P2ab Task 7 review round, reproduced twice):
- Stopword-trimmed AND rescues **1/40** (pm-vendor-chaser — the only
  query blocked solely by function words). Function words are a
  contributing factor, NOT the mechanism.
- `build_fts_match_query` reuse rescues **1/40** (kw-quillon-mast).
- OR-of-tokens: **34/40** to non-empty, **10/40** to an actual hit;
  net targets-found **20/60 → 29/60 (+10 gained, −1 lost)** — and the
  loss is `kw-plant-maintenance-record`, the golden set's only
  vector-blind fixture, whose design rests on AND-of-terms uniqueness.

## The shape: a construction sweep under a pre-registered rule

Exactly the weighting arc's proven form: enumerate candidate MATCH
constructions, run each over the full gated matrix (extending the
existing sweep machinery in `Tests/RAG_Eval/harness/fusion_sweep.py`
with a match-construction axis), and pick by a mechanical rule written
BEFORE the sweep runs. Sweep-blindness discipline (corrected at
review): the construction is CODE, not a config knob, so it cannot
enter the cache key the way rrf_k did — the sweep varies it through a
test-side injection seam and MUST rebuild the runtime or
`clear_cache()` between passes (the Task-6 counterfactual precedent),
with a sweep self-check that the control row reproduces the shipped
census (20/60) so a cache-blinded sweep cannot silently report
"construction doesn't matter". If the construction ever becomes
runtime-variable, it enters the key that day.

### Candidate constructions (the spec pre-registers these four)

1. **`and` (shipped)** — the control row.
2. **`and_stopword_trim`** — AND over content tokens only (a small
   fixed English stopword list; falls back to full AND if trimming
   empties the query). Expected ~1/40 per the attribution — included
   so the sweep RECORDS its inadequacy rather than folklore-izing it.
3. **`or`** — OR over CONTENT tokens (each still individually quoted;
   stopwords trimmed from the OR form — review finding: a raw
   OR-of-all-tokens matches every document containing "the", flooding
   negatives and precision alike; bm25's IDF discounts ubiquitous
   terms in RANKING but not in the row COUNT, and the fallback's
   entire risk surface is junk rows entering fusion). If trimming
   empties the token list, return no rows — honest, never a syntax
   error. Expected +10/−1; the −1 is disqualifying under the rule —
   included as the recall upper bound and the measured warning.
4. **`and_then_or` (the data's own suggestion)** — per sub-leg query:
   run AND first; if AND yields zero rows for that sub-leg, rerun as
   the content-token OR of construction 3. Preserves every current
   AND hit BY CONSTRUCTION (a nonempty AND never falls back —
   `kw-plant-maintenance-record`'s engine-leg resolution and hence
   its hybrid rescue stay untouched), widens only where today's
   construction returns nothing. Cost: one extra FTS query per
   zero-row sub-leg (FTS5 here is milliseconds; measure and record
   anyway). Note the deliberate mixed-mode interleave: one query may
   carry AND rows from one sub-leg and fallback-OR rows from another;
   provenance must record which rows came from the fallback so
   mechanism prose stays table-derived (the arc's prose-is-an-oracle
   lesson).

A fifth axis worth one probe each, not a full matrix row: FTS5 `NEAR`
and prefix (`token*`) variants — probe-recorded, promoted to a full row
only if a probe beats `and_then_or` on any failing category.

### Pre-registered decision rule (mechanical; the winner is computed)

- SCOPE FACT the constraints are built on (code-verified at review):
  the gated PLAIN cells ride the Library's four-seam path
  (`library_local_rag_search_service.py` ~L287/913 — "NOT the
  engine's keyword leg"), which this arc does not touch. An engine-leg
  construction change can move ONLY hybrid cells (and the leg-level
  census). The vector-blind fixture's plain rank 1 is therefore
  untouchable here, not a constraint — its LIVE protected cell is the
  hybrid rescue.
- HARD CONSTRAINTS (any violation disqualifies): (a) the vector-blind
  fixture `kw-plant-maintenance-record` keeps its hybrid rescue (for
  `and_then_or` this holds by construction; for `or` it is the
  measured loss); (b) no gated cell regresses > 0.02 in any mode
  (recall/MRR/NDCG; precision cells are gate-inert as established —
  this also covers displacement: new FTS rows entering fusion can
  displace targets sitting at ranks 9-10, exactly where the scoped
  fixtures live); (c) `Tests/RAG_Search/test_fts5_query_escaping.py`
  stays green with every candidate keeping per-token quoting.
- NEGATIVE-COMPOSITION RECORD (review finding — the gate is blind
  here, so the sweep must not be): the gated negative probes
  (`docs_at_k`, `top_score`) will NOT move under `and_then_or` for
  hybrid — the vector leg already fills k on negatives and a fallback
  FTS row's fused 0.05 cannot beat the vector rank-1's 0.7/6 ≈ 0.117 —
  but the COMPOSITION changes: fallback junk rows can enter hybrid
  top-10 at the rescue slots (~9) for absent-topic queries. The sweep
  records, per candidate, the count of FTS-only fallback rows inside
  hybrid top-10 across the 7 negatives. This is a RECORDED metric
  feeding the tie-break (fewer is better), not a hard constraint — a
  partial-match row at rank 9 of an absent-topic query is ordinary IR
  behavior, but it must be a measured, named tradeoff, never a
  surprise.
- WINNER: the candidate that, subject to the constraints, maximizes the
  count of golden queries whose target enters the keyword leg's top-10
  (the census number, 20 today); ties broken by fewest extra FTS
  queries, then by smallest code delta.
- The prompt category's gated recall-0 pin flips DISCLOSED if and only
  if the winner moves it — expected: `and_then_or` gives prompt
  queries their first rows ever.
- If NO candidate satisfies the constraints, the arc ships the sweep
  results + a re-scoped 15400 and NOTHING else — a null result is a
  recorded outcome (the pool3 precedent).

### What the winner ships through

- One construction seam: `_escape_fts5_query` grows into a
  construction-aware builder (or a sibling used by the sub-legs) — ONE
  definition consumed by all four sub-legs; the four-seam Library path
  is NOT touched (TASK-3997 owns it; cross-referenced both ways
  already).
- The construction choice is NOT a config knob in this arc (no
  unmeasured user-tunable); it is the shipped behavior, with the sweep
  row as its acceptance evidence. (If review argues for a knob, it
  defaults to the winner and the knob enters the cache key.)
- Baselines: ONE deliberate re-stamp after the winner lands (the P2ab
  discipline: intermediate runs read environment_changed only if
  fixtures change — a pure construction change flips CELLS, so the
  gate will read REGRESSION/improvement rather than
  environment_changed: the re-stamp is the arc's last act and the
  before/after table its acceptance evidence). Headroom table updates
  (prompt unblocks or its bound is re-attributed).
- TASK-15400 Done with the sweep table verbatim; 3997 gets a one-line
  note pointing at the engine-side outcome.

## Error handling

- The OR fallback inherits every sub-leg degrade path unchanged (a
  fallback query that errors degrades that sub-leg exactly as the
  primary would — no new failure modes).
- A query that is ALL stopwords (construction 2) or empties under
  tokenization falls back to the shipped AND form — never an empty
  MATCH expression (FTS5 syntax error).

## Out of scope (declared)

- TASK-3997 (four-seam AND-strictness) — pointer only.
- Semantic indexing of prompts; query expansion/PRF/HyDE (later P2c
  candidates, measured against the leg this arc ships).
- Reranking changes; fusion-parameter changes (rrf_k/alpha stay).
- New fixtures (the corpus is P2ab's; a construction change must be
  measured against the UNCHANGED instrument).

## Testing

- Always-on: construction-builder unit pins (AND unchanged for the
  shipped default; fallback triggers on zero rows only; all-stopword
  fallback; quoting preserved per candidate — extend
  test_fts5_query_escaping.py to every construction); cache-key pin
  (construction in the key); mutation checks (fallback removed → its
  pin reds; quoting dropped → escaping suite reds).
- Gated: the sweep (all four rows × full matrix); the census recount
  under the winner; the prompt-pin flip (if earned); the re-stamp.
- Live TUI: one natural-language prompt query through Library RAG
  Answer on the hybrid profile finding a prompt (the exact query class
  that returns nothing today), scratch-profile discipline as P2ab.

## Plan-phase verification items

1. The sweep machinery's extension point (how fusion_sweep parameterizes
   a strategy — add a construction axis without duplicating the runner).
2. The injection seam's exact form (constructor arg vs class attribute
   on RAGService; whether the cache is per-service-instance — if so a
   fresh runtime per pass needs no clearing at all; verify by reading
   how the service constructs its SimpleRAGCache).
3. FTS5 `rank` ordering semantics under OR (bm25 handles OR fine — but
   verify ORDER BY rank is well-defined for OR queries on all four
   sub-leg schemas, incl. prompts_fts's multi-column weighting).
4. The negative-category oracle's exact shape (results-returned@k for
   plain — constraint (d)'s measurement).
5. Timing: the per-zero-sub-leg extra query's real cost at 172 docs and
   at a realistic large library (the live check's library).
