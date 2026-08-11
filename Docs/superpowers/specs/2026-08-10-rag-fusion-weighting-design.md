# RAG Server-Port Programme — Fusion Weighting & Keyword-Leg Budget (design)

Date: 2026-08-10
Status: approved-pending-user-review
Programme: RAG server-port (P0 #1428; P1 eval harness #1458; fusion cluster #1469)
Arc: TASK-14751 (keyword-leg budget pushdown) + TASK-4110 (alpha-blend
starvation — the rescue half of TASK-3994). One branch, one PR, measured by
the P1 gate with one deliberate final re-stamp.

## Background

The fusion cluster (#1469) made hybrid genuinely fuse — and its closing task
proved, with a purpose-built vector-blind fixture, what it could not fix:

- **TASK-4110 (high):** an FTS-only row — a document the keyword leg found
  and the vector leg did not return — cannot enter hybrid's top-k. Fused
  score `(1-α)/(rrf_k + fts_rank) + 0`; at the shipped defaults (α 0.7,
  rrf_k 60) FTS rank 1 scores 0.3/61 = 0.00492, beaten by every vector row
  ranked better than ~82, while `_hybrid_search` asks the vector leg for
  only `top_k × 2`. Measured: `kw-plant-maintenance-record` → plain rank 1,
  semantic absent, engine FTS leg rank 1, hybrid **absent** (sorts 21st
  behind 20 vector rows). Feeding the same function a wider pool merged the
  target at vector rank 22 and landed it at fused rank 1 — a
  scoring/weighting property, not a wiring bug.
- **TASK-14751 (high):** TASK-3996 round-robins media/notes/conversations
  into one fixed top_k FTS budget while the Library post-filters by selected
  source types AFTER fusion. Reviewer probe: 12 matching docs of each type,
  leg asked for 20 → {media 7, note 7, conversation 6}; with Media-only
  selected, 13 of 20 rows are discarded. Worst case (media-only + thin/empty
  vector index — the "keyword-only results" case) the user sees ~⅓ of the
  media rows dev previously returned. Invisible to the eval gate (harness
  selects all three types) and to Library unit tests (canned fakes).

Both task files carry full ACs; they are the per-task contracts.

## Diagnosis that shapes the design (verified arithmetic)

**The starvation lever is `rrf_k`, not alpha.** With rrf_k 60 over a ~20-row
candidate window, the RRF curve is nearly flat: vector rank 1 → 1/61 =
0.0164, rank 20 → 1/80 = 0.0125. The spread across twenty ranks (~0.004) is
smaller than the leg-weight gap (0.7 vs 0.3 ≈ factor 2.3), so **leg
membership dominates relevance rank**: FTS rank 1 loses to vector rank 80.
rrf_k = 60 is the server's constant, calibrated for candidate pools of
thousands; chatbook applies it to pools of ~20.

Consequences for the option space TASK-4110 lists:

- **Per-leg RRF renormalization is a structural no-op** — a shared
  `1/(k+1)` normalizer cancels out of every pairwise comparison. Ruled out
  by arithmetic; stated here so it is not re-explored.
- **Retuning alpha alone** needs α < 0.567 before FTS rank 1 beats vector
  rank 20 — a global penalty on vector quality to fix an edge case. Wrong
  lever alone; alpha stays a candidate only in combination.
- **Lowering rrf_k** is the proportionate lever: at k = 10, FTS rank 1
  scores 0.3/11 = 0.0273 and beats vector rows past rank ~16, and rank
  ordering regains weight *within* each leg.
- **Widening the vector candidate pool** (multiplier 2 → larger) fixes the
  common near-miss case (doc present below 2k) but not genuine
  vector-absence; cheap, likely a companion to whichever weighting wins.
- **A keyword slot quota** (guarantee the FTS leg's best row(s) N of the
  top-k slots) is the only option that guarantees rescue under genuine
  vector-absence regardless of weights; its risk is admitting junk when the
  keyword leg's best is weak.

**The decision is made by measurement, not argument** (TASK-4110 AC#2). The
arc builds a small strategy-comparison harness on P1's machinery, scores the
candidates over the golden set, and ships the winner with its numbers.

## Sequencing (load-bearing)

1. **TASK-14751 first.** It is the live regression risk, it is independent,
   and fixing the budget split before measuring means TASK-4110's numbers
   describe a keyword leg that is not wasting two-thirds of its slots.
2. **The comparison harness second**, on the corrected leg.
3. **TASK-4110's chosen fix third**, with the measurement in hand.
4. One deliberate final re-stamp + closure (TASK-3994 `#2b` ticked at last).

## Fix designs

### TASK-14751 — source-type pushdown into the keyword leg

- `RAGService.search`/`_hybrid_search`/`_keyword_search` accept an optional
  `keyword_source_types: Collection[str] | None` (None = all three — every
  existing caller unchanged). The Library's hybrid arm passes the user's
  selected types translated to the engine vocabulary (`media` / `note` /
  `conversation`), translation pinned by test (the singular-vocabulary trap
  from the fusion cluster).
- The leg runs ONLY the selected sub-legs and interleaves rank-fairly among
  them; a single-type selection degenerates to that sub-leg's natural
  best-first order with the FULL top_k budget (AC#2: media-only returns as
  many media rows as the pre-3996 leg did).
- Rank-fair interleaving among multiple selected types is preserved (AC#4 —
  the trade-off being restored, not reverted: FTS5 scores are not comparable
  across tables, so concatenation would let one well-stocked source consume
  every slot).
- Composition pinned against a REAL mixed corpus (AC#3, the
  test_keyword_leg_chacha.py pattern): reds if the budget silently reverts
  to a fixed three-way split under a single-type selection.
- Scope note: the plain-profile route and Library `search` mode use the
  four-seam Library path, not this leg — untouched.

### The strategy-comparison harness (measurement, then deletion of losers)

- A gated comparison runner beside the P1 harness (Tests/RAG_Eval/harness/,
  env-gated like everything else there) that runs the golden set through
  hybrid under parameterized fusion settings and reports per-category
  metrics + the rescue verdict for `kw-plant-maintenance-record` per
  strategy:
  1. baseline (α .7, k 60, ×2) — the control;
  2. lowered rrf_k (a small sweep, e.g. k ∈ {5, 10, 20});
  3. widened pool (multiplier ×2 → ×5 and/or a fixed floor);
  4. keyword slot quota (best FTS-only row guaranteed one slot when its
     leg-rank is strong; exact rule parameterized);
  5. the most promising combination(s).
- The runner is a MEASUREMENT TOOL: its full matrix runs in the arc to make
  the decision; what ships permanently is the winner's configuration + a
  slim always-on regression pin, not a combinatorial test suite. The runner
  itself stays (gated) for future retunes — it is small and rides P1's
  existing runtime.
- **Parameterization prerequisites (code-verified, second review):**
  - `rrf_k` is hard-coded at the engine call site
    (`_fuse_hybrid_results` passes `rrf_k=DEFAULT_RRF_K`) while
    `fusion.resolve_rrf_k` — the validator for a config knob — already
    exists, unconsumed. Before the harness can measure k variants, add
    `config.search.rrf_k` (validated via `resolve_rrf_k`, default 60) and
    make the call site read it. One authoritative site; it doubles as the
    shipping site if k wins. Same treatment for the pool multiplier
    (`SEARCH_RESULT_MULTIPLIER` is a module constant) if pool widening is
    measured: a config field with the current default.
  - **Metadata-honesty trap:** the `hybrid_fusion` metadata block records
    the alpha/rrf_k used, and P1's citation capture (`_reliable_rrf`)
    RE-DERIVES the fused score from those recorded values — threading a
    config k without recording the ACTUAL value silently degrades every
    hybrid row to the LEGACY score-kind (the exact silent-failure class
    the P1 arc fixed once already). The block must record the values
    actually used; the re-derivation then doubles as an arithmetic guard
    that any k/alpha drift breaks loudly. A test pins metadata-recorded k
    == configured k.
  - **Measurement cost asymmetry (YAGNI ordering):** k, alpha, and
    multiplier variants are parameter sweeps over existing code once the
    knobs exist; the slot quota requires BUILDING a mechanism first. The
    harness measures the parameter strategies first; the quota mechanism
    is implemented only if no parameter strategy satisfies the decision
    rule.
- Decision rule, stated before measuring (no post-hoc goalpost moves).
  Two distinct rescue senses, both required (self-review catch — the
  fixture doc is vector-POOR, at vector rank ~22 below the 2k cutoff, not
  vector-ABSENT, so pool widening alone would move AC#3's cell while
  leaving AC#4 structurally unsatisfiable):
  - **AC#4's structural guarantee** — an FTS-only row outranks at least one
    vector-only row under the SHIPPED defaults — is satisfiable only by a
    weighting change (rrf_k, quota, or an alpha combination). The winner
    MUST include one; pool widening is a companion candidate, never the
    sole winner.
  - **AC#3's fixture outcome** — `kw-plant-maintenance-record` miss → hit —
    may be achieved by the weighting change, the widened pool (merge-rescue
    at vector rank 22), or both; the PR states which mechanism did it.
  The winner must also (b) not regress any per-category recall/MRR/NDCG
  cell by more than the gate's warn band (0.02), and (c) among qualifying
  strategies prefer the smallest deviation — tie-break order among
  weighting levers: rrf_k, quota, alpha-combo; pool widening added when it
  independently earns its keep. Precision cells are expected to move
  mechanically with row-count changes (the fusion cluster's lesson) and are
  reported but not disqualifying, with the mechanical share explained.
- **ADR-005 note:** the k=60/α=0.7 defaults came from the server-parity
  commitment (`backlog/decisions/005-invest-in-local-rag-mirroring-tldw-
  server.md`). The server's constants are calibrated for candidate pools of
  thousands; chatbook's pools are ~20. Deviating is a conscious,
  measurement-backed refinement of that ADR's intent (mirror the DESIGN,
  not constants tuned for a different regime) — the PR says so, and the
  ADR gets a one-line addendum rather than silent divergence.
- If NO strategy satisfies (a)+(b): STOP; report the matrix; the owner
  chooses the trade-off. That is a finding, not a failure.

### TASK-4110 — ship the winner

- Implement the winning configuration in `RAG_Search/fusion.py` /
  `rag_service.py` config (whichever the winner touches), with the config
  surface documented (profile knobs stay honest: if rrf_k changes, the
  Settings' hybrid documentation and any surfaced constants follow).
- **Shared-blend guard:** `pipeline_builder_simple` consumes the same
  `reciprocal_rank_fusion` (TASK-3501, legacy path). The winner must either
  leave the shared function's defaults untouched (preferred: parameterize at
  the caller) or, if a default changes, say so explicitly in the PR and
  leave a note on TASK-3501. No silent behavior change to the legacy path.
- AC#4's regression pin: an FTS-only row outranks at least one vector-only
  row under the SHIPPED defaults — always-on, hand-built rows.
- AC#5's fixture guard: a gated test asserts semantic mode still does NOT
  return `note-saltmarsh-hide` for `kw-plant-maintenance-record` (the
  corpus stays able to distinguish coverage from noise across future model
  bumps/re-stamps).
- TASK-3994 `#2b` ticked with the rescue evidence; 4110's own ACs closed.

## Error handling

- No new failure modes: pushdown with an empty selection is unreachable
  (the Library gate already routes no-keyword-source selections to
  semantic); defensively, an empty `keyword_source_types` collection
  behaves as "no sub-legs" (leg returns [], hybrid degrades to semantic —
  the existing disclosed path).
- The comparison runner inherits P1's skip conditions (env gate, extras,
  model cache) — never a new gating mechanism.

## Testing

- RED-first per behavior; mutation checks on the pushdown (drop it → the
  composition test reds) and on the shipped weighting (revert to old
  defaults → AC#4's pin reds).
- P1 gate: informational runs mid-arc; ONE deliberate final re-stamp with
  the full delta printout and a per-stage progression table (baseline →
  post-14751 → post-4110-winner = stamped). 14751 is expected to move
  nothing on the gate (the harness selects all three types — its fix is
  gate-invisible by construction, covered by unit tests instead; the spec
  says this so nobody reads +0.000 as evidence of anything).
- Protected oracles: the fusion cluster's tests (doc fusion, chacha leg,
  fts5 escaping, private-sqlite inventory) pass UNMODIFIED except where a
  changed shipped default legitimately moves a pinned constant — any such
  edit is disclosed as a conscious oracle update, never silent.
- Live TUI check in the closing task: the vector-blind scenario on real
  data — a query whose answer exists only as an exact keyword match appears
  in hybrid results (the user-visible rescue), scratch profile, PR-2 recipe.
- Collection arithmetic; final whole-branch review; targeted suites PLUS
  `Tests/DB/test_private_sqlite_inventory.py` (the guarded-KIND lesson —
  this arc touches no connections, but the leg orchestration moves code
  near one; cheap insurance).

## Out of scope (declared)

- TASK-3501 (legacy pipeline_builder fusion twin) — noted at the shared
  seam, not unified here.
- TASK-14752 (coverage-copy honesty), TASK-4111 (Library Open), TASK-3997
  (AND-vs-OR strictness) — separate arcs.
- P2 feature ports (expansion/HyDE/PRF etc.) — after this lands.
- Four-seam engine keyword parity for prompts; scope-aware hybrid
  (allowlists) — P2.

## Plan-phase verification items

1. The exact signature threading path for `keyword_source_types`
   (Library hybrid arm → RAGService.search → _hybrid_search →
   _keyword_search) and which existing callers pass positionally.
2. Whether `interleave_rankings` handles a single-ranking input degenerately
   (it should — verify, don't assume).
3. The comparison runner's cheapest honest form: reuse `run_eval`'s
   per-mode machinery with a parameterized service config, or a narrower
   hybrid-only loop; pick whichever avoids duplicating the report format.
4. RESOLVED at spec time (second review): alpha is config-threaded
   (`config.search.hybrid_alpha` via `resolve_hybrid_alpha`); rrf_k is
   hard-coded at the call site with `resolve_rrf_k` existing unconsumed;
   `SEARCH_RESULT_MULTIPLIER` is a module constant. The plan adds
   `config.search.rrf_k` (and, if measured, a multiplier field) before the
   harness measures. Remaining to verify: every OTHER `DEFAULT_RRF_K` /
   `SEARCH_RESULT_MULTIPLIER` import site (the semantic leg also uses the
   multiplier — a pool-widening knob must widen the HYBRID legs without
   silently changing semantic-mode behavior).
5. TASK-3994 `#2b`'s exact wording so the closing tick quotes its own
   criterion.
