# RAG P2c feature admission — PRF, probed before built (design)

Date: 2026-08-13
Status: draft-pending-user-review
Programme: RAG server-port (P0 #1428; P1 #1458; fusion #1469; weighting
#1487; P2ab #1517; 15400 #1574; 15700 #1608 — dev `0d718e7fb`)
Arc: the first P2c FEATURE candidate (PRF — pseudo-relevance feedback),
run under the discipline the instrument has now earned four times over:
**the premise is probed and priced before any production code exists**,
with pre-registered admission criteria and a pre-registered kill
condition. Three P2c premises (expansion, acronym, compositional) died
under exactly this scrutiny; PRF gets no exemption for being next in the
approved cost order.

## The premise, stated honestly before the probe

PRF's classic mechanism: take the top-k results of a first-pass
retrieval, extract their most characteristic terms (RM3-style TF or
TF-IDF over the pseudo-relevant set), and re-run the query expanded with
those terms. It is a LEXICAL-bridging device — its historical value is
letting a keyword engine reach documents that share vocabulary with the
query's best matches rather than with the query.

Where that CAN pay on this corpus, and where it structurally cannot:

- **Plain profile (BM25 Only users), paraphrase + vocabulary_mismatch:
  the honest target.** These cells sit at 0.000 recall — 22 queries the
  four-seam keyword path cannot answer because the query shares no
  content word with its target. PRF is the textbook fix for exactly this
  population... IF the first pass retrieves anything to feed from. THE
  PROBE'S CENTRAL QUESTION: for how many of those 22 queries does the
  first pass return pseudo-relevant material whose expansion terms reach
  the target? (A first pass returning zero rows feeds PRF nothing; a
  first pass returning topically-wrong rows feeds it poison.)
- **Hybrid, negation (the only clean gated headroom): PRF is expected to
  be USELESS-OR-HARMFUL, pre-registered.** Negation queries fail because
  the corpus asserts what the query excludes; expanding the query toward
  the top results' vocabulary drags it FURTHER toward the assertions.
  The probe measures this anyway — a pre-registered expectation is not
  evidence — but the admission bar treats negation as a REGRESSION GUARD
  (PRF must not add junk rows to negation results), never as its target.
- **Hybrid, everything else: at ceiling or bounded by fusion.** Vector
  retrieval already answers paraphrase/vocab at 1.000; prompt movement
  is bounded by the fusion boundary (the and_then_or (b)-failure
  mechanism), not by query formulation. No hybrid admission claim will
  be entertained from the probe unless a cell that is NOT at ceiling
  moves without any cell regressing — the standing (b) constraint.

## Phase A — the probe (no production code)

A harness-side probe (`Tests/RAG_Eval/harness/` — the fixture_probe /
sweep idiom; gated, RAG_EVAL=1) implementing the SIMPLEST honest PRF:

- **Step 0 — the fireability census (review addition; run FIRST, it is
  one command):** for each of the 22 failing queries, does the shipped
  first pass return ANY rows? The four-seam builder is AND-strict across
  terms (verified: `build_fts_match_query` = AND of plural/singular
  OR-groups), so a paraphrase query's first pass plausibly returns zero
  rows — and a zero-row first pass feeds PRF NOTHING, structurally. If
  fireability < 5/22 on the shipped first pass, ONE pre-registered
  variant is licensed: the first pass run in an OR-of-content-terms form
  FOR FEEDBACK ONLY (users would still see the shipped results; the
  wider pass exists only to select pseudo-relevant documents — classic
  PRF on a candidate set). No other variant is licensed; if BOTH
  fireability regimes null, the arc records the null. Pre-registering
  this now is what keeps it from being post-hoc scope creep later.
- **Row-content fact the probe must design around (verified at review):
  four-seam media and conversation rows carry NO document text** — their
  snippets are labels ("Matched media · {type}", "Matched conversation ·
  N messages"); only note and prompt rows carry real text. Term
  derivation therefore requires a content fetch for the top-M fed rows
  via the existing read APIs (one read per fed row — priced and
  reported), or the feed is silently biased toward notes/prompts. The
  probe fetches; the price rides the report.
- First pass: the leg the profile would run — composed at the DB-level
  call sites the four-seam service itself uses (the seams build a MATCH
  string via `build_fts_match_query` and hand it to
  `db.search_*_by_content`-family functions, so the probe can hand its
  own expression AT THE SAME CALL SITES and stay on the product's SQL;
  the 15400 probe-fidelity lesson — measure the form the engine would
  actually run).
- Term derivation, pre-registered: top-N terms by TF over the top-M
  first-pass documents (N=8, M=5 to start — swept over a SMALL
  pre-registered grid {N: 4/8/16, M: 3/5/10} only if the base point
  shows signal; a null at every base point is a null, not an invitation
  to search the grid until something moves — record every point run).
  Stopwords excluded via the existing `_FTS5_STOPWORDS`; terms already
  in the query excluded.
- Second pass: the query's content terms OR-extended with the expansion
  terms — via the EXISTING construction seam (an expression handed to
  the leg, not a new construction; the probe may compose expressions the
  way the sweep's probes did). Per-token quoting preserved (the
  injection property is load-bearing in probes too).
- Measured, per query in the target population (the 22 plain-failing +
  the 3 negation as guard + the 7 negatives as junk-guard): target
  reached (rank), rows returned, junk delta on negatives/negation.
  Plus the OLD-QUERIES guard: **every plain query currently hitting its
  target — derived from a fresh baseline pass at probe time, NOT a
  hardcoded list** (review correction: plain keyword recall is 0.844,
  not 1.000, and the scoped category's 7/7 plain hits are in the guard
  population too) — must not lose its target to expansion-term dilution
  (the 15700 lost-column discipline: gains AND losses by query id).

### Pre-registered admission bar (Phase B exists only if met)

- ≥5 of the 22 plain-failing queries reach their target in the second
  pass's top-10, AND
- zero currently-hitting plain queries (any category) lose their
  target, AND
- zero new rows on negatives. **Honesty note (review): under the
  shipped AND-strict first pass this guard is STRUCTURAL — negatives
  return zero rows, so PRF cannot fire on them and the guard cannot
  bind; it becomes a REAL guard only under the OR-feedback variant,
  which is exactly when it matters. Report it as structural-vs-live
  accordingly (a guard that cannot bind is a property of the first-pass
  shape, not evidence of safety).** AND
- the negation guard: no negation query's row set grows with
  assertion-side junk (measured, reported; expected to bind).
- A result below the bar in every grid point = THE NULL: recorded in the
  README beside the retired premises (expansion/acronym/compositional),
  TASK for the next candidate (clarification gate) filed with the probe
  machinery pointer, arc ends WITHOUT production code. The null is a
  success outcome of the discipline, exactly like pool3 and the 15400
  OR rows.

## Phase B — build only what Phase A admitted (conditional)

If the bar is met: PRF ships as a PLAIN-PROFILE-ONLY second-pass option
behind the profile system (a `SearchConfig`/profile field, OFF by
default on hybrid/semantic profiles — the probe's evidence licenses
plain only), through the four-seam path's seams. Design constraints that
bind Phase B (pre-stated so the plan cannot drift):

- The expansion is DISCLOSED in the route-note vocabulary ("expanded
  with N terms from first-pass results" — the honesty conventions the
  Library surface already carries).
- The second pass is one extra leg query per source type (priced and
  reported like 15700's 220 statements; wall-time measured).
- The gated instrument gains the plain-PRF cells via ONE deliberate
  re-stamp; hybrid/semantic cells must read +0.000 (PRF is off there —
  the zero-movement proof).
- 3997 (four-seam AND-strictness) interaction: the expanded second pass
  is OR-shaped by construction; it does NOT modify the first pass or
  3997's surface — a note on 3997, no scope creep.
- Live check: a BM25 Only profile user's paraphrase query finding its
  target, scratch-profile discipline; the RAG-Answer UI hazard
  (TASK-15810) budgeted per the 15700 precedent.

## Out of scope (declared)

- HyDE, clarification gate, granularity router (later candidates; each
  gets its own probe when its turn comes).
- Any hybrid/semantic-profile PRF (the probe's evidence cannot license
  it; a future arc may re-probe post-fusion-boundary work).
- Fusion parameters; the keyword-leg construction (shipped and stable);
  new fixtures (the corpus is the instrument — a probe that needs new
  fixtures to show value is measuring the fixtures).

## Testing

- Phase A: the probe is gated harness code with always-on pure-function
  tests for term derivation (stopword/query-term exclusion, N/M cuts,
  determinism) and expression composition (quoting, OR shape); the probe
  REPORT carries the per-query table verbatim.
- Phase B (conditional): RED-first through the four-seam path's existing
  test patterns; the disclosure copy pinned; the off-by-default pinned
  per profile; mutation on the admission-scoped gating (PRF firing on a
  hybrid profile → its pin reds); ONE re-stamp; the environment
  discipline (worktree venv --python 3.12 + pinned fingerprint packages;
  import provenance asserted before every measurement; the re-stamp in
  the fingerprint-matching main venv).

## Plan-phase verification items

1. The four-seam path's second-pass seam: where a probe (and later a
   feature) can hand an expanded expression per source type
   (`library_fts_query.build_fts_match_query`'s consumers — read the
   seams at `library_local_rag_search_service.py:494/528/548/586`).
2. Whether the four-seam leg exposes enough of its first-pass rows
   (content text) for term derivation without a second content fetch.
3. The profile system's surface for a plain-only flag (config_profiles
   — how BM25 Only / plain profiles would carry it; NOT TOML-wired
   beyond the profile, mirroring the construction seam's precedent).
4. The negation guard's oracle shape (rows-returned + assertion-side
   containment — what "assertion-side junk" is measurable as).
5. Grid runtime at 172 docs (the probe is 32 queries × ≤9 grid points ×
   2 passes — bound it before running).
