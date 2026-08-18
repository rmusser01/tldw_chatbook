# RAG_Eval — the P1 retrieval eval harness

This directory is the measuring instrument for `tldw_chatbook`'s Library
retrieval seam (`LibraryLocalRagSearchService`). P0 (task-3170) made
profile-driven retrieval reachable from Console and Library search but
deliberately made no quality claims. P1 (TASK-3894) builds the thing that
can make those claims: ported precision/recall/MRR/NDCG/F1@k metrics, a
regression/gating layer with fingerprinted committed baselines, a
deterministic fixture corpus + golden query set spanning notes, media,
conversations and saved prompts, and an env-gated pytest harness that runs
every golden query through the real seam across all three profile modes.

> **The instrument was renewed in P2ab (TASK-15020, 2026-08-11).** 172
> documents, 60 queries, with three fail-first classes — `scoped`,
> `negation`, `prompt` — admitted only where today's pipeline was *measured*
> to fail them. Hybrid overall recall came off the 1.000 ceiling it had sat
> at since the weighting arc and now reads **0.848** (0.826 at the end of
> P2ab; TASK-15400's construction flip took it the rest of the way), and the
> baselines were re-stamped once, deliberately, at the end of each arc that
> moved a cell — TASK-15700 (2026-08-13) moved the default a second time and
> moved **none**, so it re-stamped nothing and said so. **TASK-17755
> (2026-08-18) re-stamped `plain.json` and only `plain.json`**: adopting the
> engine's `and_then_prefix` construction on the Library's four-seam keyword
> path took plain overall MRR/precision 0.304 → 0.326 and the `keyword`
> category 0.875 → 0.938 (all ten plain cells up; `mean_docs_at_k`
> 0.304 → 0.457), while every `semantic` and `hybrid` cell held at +0.000 —
> that mode split is the evidence the change is confined to the path it
> claims. Two candidate
> classes (`compositional`, `acronym`) proved **unfailable** on this corpus
> and model and were not authored — that is recorded evidence against two
> P2c feature premises, not an omission. A **fourth** premise died the same
> way in TASK-15965 (2026-08-13): pseudo-relevance feedback was probed
> before it was built and came back NULL — see "The fourth retired P2c
> premise" under the admission protocol. A **fifth** died on 2026-08-18
> (TASK-16072): the **clarification gate**, killed by a CENSUS rather than a
> probe — of 60 golden queries only 2 have more than one relevant document,
> and both of those have two CORRECT answers, so a clarifying question would
> ask the user to discard a right one. Qualifying queries: 0 against a
> pre-registered bar of 5. The census cost one query over the fixture and
> reached the same kind of answer PRF needed a full probe for, which is why
> the next candidate (TASK-18155, granularity router) requires a census
> first. Start at the **headroom table**
> below: it names, per category, what is left to improve and what can only
> be regressed.

**Why this exists.** Every later retrieval change in the RAG-port programme
(P2's query expansion, HyDE, PRF, a clarification gate, a granularity
router; any threshold or model tuning) must show its effect as measured
numbers against these committed baselines, run locally — CI is intentionally
dead in this repo, so the gate is a local pytest run before you open the PR.

## Layout

```
Tests/RAG_Eval/
  test_metrics.py                  always-on: precision/recall/MRR/NDCG/F1@k known-answer tests
  test_regression_gating.py        always-on: ported regression + gating module tests
  test_goldenset_integrity.py      always-on: fixture loader + validator tests
  test_canonicalize.py             always-on: row -> doc-id canonicalization tests
  test_baseline_io.py              always-on: baseline round-trip + gate-arithmetic tests
  test_runner_error_paths.py       always-on: three-mode runner's error/edge paths (fake seam, no model)
  test_harness_smoke.py            env-gated: harness can stand up a real indexed runtime at all
  test_harness_run.py              env-gated: the real three-mode run, and the baseline gate itself
  test_harness_scoped.py           env-gated: what a SCOPED query routes to today (the before-pin)
  test_prf_probe.py                always-on: the PRF probe's pure functions (28 pins)
  test_prf_probe_run.py            env-gated: the PRF probe RUN — census, oracle controls, grid, guards, verdict
  conftest.py                      un-sandboxes the model cache dir for env-gated tests only
  harness/
    environment.py                 the RAG_EVAL gate + HF-offline latch
    goldenset.py                   fixture loader/validator (corpus.toml, golden.toml)
    ingest.py                      corpus -> real source DBs -> isolated indexed RAGService
    canonicalize.py                seam rows -> fixture-slug document ids
    runner.py                      the three-mode run + per-category scoring
    baseline_io.py                 fingerprinted baselines + the fail-on-regression gate
    prf_probe.py                   FEATURE-PROBE machinery: term derivation + expression composition (pure)
  fixtures/
    corpus.toml                    172 fixture documents (note/media/conversation/prompt)
    golden.toml                    60 golden queries (46 scored, 7 negative, 7 scoped)
  baselines/
    semantic.json, plain.json, hybrid.json   committed, fingerprinted, per-mode baselines
```

The metrics and regression/gating modules live in
`tldw_chatbook/RAG_Search/eval/`, ported from tldw_server2's `rag_service`.
Fixture-integrity (`goldenset.py`) and canonicalization are chatbook-native
harness code, not ports. All four are always-on: they need no model, no
extras, and run in every ordinary `pytest` invocation.
Only the files that stand up a real indexed corpus and run real queries
through the real seam (`test_harness_smoke.py`, `test_harness_run.py`,
`test_harness_scoped.py`, `test_fusion_sweep.py`) are gated behind
`RAG_EVAL=1`.

## Running it

**Always-on (no gate, no model, part of the normal suite):**

```bash
pytest Tests/RAG_Eval/ -q
```

This runs the metrics/gating/fixture-integrity/canonicalization tests —
**measured 2026-08-13: 266 passed, 12 skipped** (the skips are the env-gated
files' tests). It never touches the embedding model or a real corpus.

**The real harness run (opt-in, slow, needs a real model):**

```bash
RAG_EVAL=1 pytest Tests/RAG_Eval/ -q -p no:randomly
```

This runs everything above *plus* `test_harness_smoke.py` and
`test_harness_run.py`, which stand up a genuine indexed RAG install (the 172
fixture docs, written through the real writer APIs, embedded and indexed
through the real batch path) and run all 60 golden queries three times —
once per `default_search_mode` (`plain`, `semantic`, `hybrid`). Whole run is
~10-15s once the model is warm. `-p no:randomly` is not required (the run
passes in either order — see Task 7) but keeps the two env-gated tests'
output adjacent when you are reading it.

The gate skips itself, with a reason, when:

- `RAG_EVAL` is not `"1"` — the default, so an ordinary `pytest`/`pytest
  Tests/` run is never slowed down by this;
- the `embeddings_rag` extras are not installed
  (`pip install "tldw_chatbook[embeddings_rag]"`);
- the `all-MiniLM-L6-v2` embedding model is not already in the local
  HuggingFace cache. **The harness never downloads a model** — `HF_HUB_OFFLINE`
  is enforced for the duration of a gated run (see `harness/environment.py`
  for exactly where that has to happen; it is earlier than "before import").
  If the model is not cached, pre-fetch it once outside the harness, or
  point `TLDW_RAG_EVAL_MODEL_CACHE` at a cache directory that has it.

**Updating (re-stamping) the baselines:**

```bash
RAG_EVAL=1 RAG_EVAL_UPDATE_BASELINES=1 \
  pytest Tests/RAG_Eval/test_harness_run.py::test_the_committed_baselines_still_hold -q -s
```

Writes `Tests/RAG_Eval/baselines/{semantic,plain,hybrid}.json` and prints
every one of the 105 gated metrics old -> new (or `absent -> value` on first
stamp) before you commit — the point is a reviewable diff, never a silent
overwrite. Read the printed deltas before committing; a re-stamp that moves
every hybrid number by the same amount is a defect fix or a config change,
not noise.

## The P2 discipline: only re-stamp deliberately

The baseline files are checked in. **Do not re-run with
`RAG_EVAL_UPDATE_BASELINES=1` because the gate failed** — a failure means
either a real regression (fix the regression) or a deliberate, understood
change (a retrieval algorithm change, a threshold tune, a corpus edit). Only
in the second case do you re-stamp, and when you do:

1. Run the gate once *before* your change (or read the committed baseline
   numbers directly — they are plain JSON) to get the "before" figures.
2. Make your change, re-stamp, and capture the printed "after" deltas.
3. Put **both** sets of numbers in the PR description, not just the fact
   that the baseline changed. A baseline-only diff with no numbers in the PR
   body is not reviewable.
4. Never re-stamp and land the code change in separate PRs — a baseline
   that does not match the code it was measured against is worse than an
   absent one, because it looks authoritative.

**This happened for real, twice: the fusion cluster (TASK-3995 / 3994 /
3996 + a corpus addition), re-stamped in one commit, and then the
weighting arc (TASK-14751 + TASK-4110), re-stamped in one commit.** The
first baselines encoded a defect — hybrid search was byte-identical to
semantic search on this corpus (44/44 identical id-lists), because the two
legs fused on mismatched id spaces. The progression, each step measured by
a gated run against the *then-current* committed baselines, with only the
two marked steps re-stamped:

| step | hybrid overall P / F1 | hybrid keyword P / R | note |
|---|---|---|---|
| P1 baseline (defect) | 0.117 / 0.208 | 0.135 / 1.000 | hybrid ≡ semantic, 0/44 queries differ |
| + TASK-3995 (per-token FTS quoting) | 0.117 / 0.208 | 0.135 / 1.000 | +0.000 on all 60 gated metrics — the engine's keyword leg was reached but its rows could not yet surface |
| + TASK-3994 (fuse on document identity) | 0.105 / 0.190 | 0.113 / 1.000 | 22/44 queries now differ from semantic; real merged rows appear |
| + TASK-3996 (notes/conversations sub-legs) | 0.105 / 0.190 | 0.113 / 1.000 | +0.000 again: every query whose FTS leg fires already had its target at vector rank 1 |
| + vector-blind fixture (**stamped**) | 0.103 / 0.185 | 0.106 / 0.938 | corpus 48→49 docs, golden 44→45; the new query is the first one hybrid *cannot* answer |
| + TASK-14751 (source-type pushdown) | 0.103 / 0.185 | 0.106 / 0.938 | +0.000 on all 60 gated metrics, **expected by construction**: the harness always selects all three keyword-indexed sources, so pushing that selection into the FTS leg cannot change what the leg searches here. The fix shows in product behaviour (a Notes-only search stops searching media), not in these numbers. |
| + TASK-4110 (`rrf_k` 60 → 5, **stamped**) | 0.105 / 0.190 | 0.113 / 1.000 | the keyword leg's unique find finally survives fusion: `kw-plant-maintenance-record` goes miss → **hit at rank 8, FTS-only**. Every moved cell is a rise; `semantic` and `plain` are +0.000 on all 40 of their gated metrics, which is the check that the change touched hybrid fusion and nothing else. |

Three things in that table are worth more than the numbers. First, **three
of the five fixes moved nothing**, and that is a finding rather than a
disappointment: it says this corpus could not tell coverage from noise,
because semantic recall@10 was 1.000 everywhere and the FTS leg's finds
were always documents the vector leg had already ranked first. Second, the
vector-blind-fixture row moved the numbers **down**, because the corpus
finally contained a document the vector leg misses — and hybrid missed it
too, which is what made TASK-4110 measurable at all. Third, the last row is
the only one where a *weighting* change (not a wiring change) moved a
number, and it moved exactly one query: keyword recall 0.938 → 1.000 is
15/16 → 16/16.

### The headroom table (what P2c can actually move)

**The ceiling is gone.** At the end of the weighting arc every scored query
in this corpus was a hit in hybrid mode (overall recall 1.000), so a fusion
retune had nothing left to improve here and could only be measured for
regression. P2ab's fail-first authoring is what fixed that. These are the
numbers the re-stamp committed (`k=10`, 60 queries, 46 scored — negatives
and scoped are excluded from the overall row and reported in their own
cells):

| category | n | semantic R@10 | plain R@10 | hybrid R@10 | headroom |
|---|---|---|---|---|---|
| `keyword` | 16 | 0.938 | 0.844 | **1.000** | hybrid at ceiling; semantic/plain can still show a rise |
| `paraphrase` | 13 | 1.000 | 0.000 | 1.000 | **none in the vector modes** — regression-only |
| `vocabulary_mismatch` | 9 | 1.000 | 0.000 | 1.000 | **none in the vector modes** — regression-only (see the caveat under Category meanings) |
| `negation` | 3 | 0.000 | 0.000 | **0.000** | **full** — nothing retrieves these today |
| `prompt` | 5 | 0.000 | 0.000 &nbsp;⚠️ | **0.200** | ⚠️ **the `plain` cell is VACUOUS, not measured** — the harness leaves `prompt_scope_service=None` so that seam reports itself unavailable (TASK-18255); read only the `semantic`/`hybrid` cells as retrieval. 1 of 5. TASK-15400's construction flip took it; TASK-15700's merge fix + `and_then_prefix` flip held the cell at 0.200 while the MECHANISM under it moved (stopword trim → prefix fallback). The residual 4 are bounded by absent CONTENT words — see below |
| `scoped` | 7 | 0.000 | 1.000 | **1.000** | hybrid flipped from 0.000 in this arc (B1); MRR 0.163 is the remaining headroom, not recall |
| **overall** | **46** | **0.804** | **0.293** | **0.848** | hybrid is **0.152** off the ceiling |

**The remaining 0.000 rows are P2c's admission targets.** `negation` (all
three modes) and `prompt` (0.000 in `semantic`; `plain`'s 0.000 is **vacuous** — see the
⚠️ note) are the cells with room to rise, and they are not the same kind of
problem:

- **`negation` 0.000 in all three modes is a genuine open capability gap.**
  Three fixtures, each describing the exception *without ever naming the
  aspect the query negates*, so the cue word lives only in the
  norm-asserting documents: the keyword paths cannot reach the target and
  the vector leg is pulled onto the norm. Nothing in the pipeline addresses
  this today. This is the cell to move.
- **`prompt` hybrid 0.200 is one query, and the other four are an HONEST
  BOUND rather than a B2 failure.** B2 shipped the prompts keyword sub-leg
  and it is **proven reachable end-to-end** — the same runtime answers
  "shift log summary supervisor" with the right prompt at hybrid rank 9,
  FTS-only, provenance read off `metadata`. The category read 0.000 in all
  three modes until TASK-15400 measured four MATCH constructions and shipped
  `and_stopword_trim`, which drops function words from the AND: that rescued
  exactly one golden query (`pm-vendor-chaser`, blocked solely by "about")
  and moved the hybrid cell to 0.200. **The other four still miss in every
  mode, and not for a reason a stopword list can fix** — they miss on absent
  CONTENT words (`template`, `building`, `rough`, `turns`, `pulls`,
  `builds`). TASK-15700 (2026-08-13) then fixed the merge, added
  plural/singular widening as a **fallback** (`and_then_prefix`), and the
  cell **still reads 0.200** — the same one query, now reached by the prefix
  fallback rather than by the trim. So the residual four are bounded by
  absent content words, full stop; the two constructions that DO answer all
  five (`or`, `and_then_or`: prompt census 5/5) were re-measured under the
  fixed merge and are **still disqualified, for two different reasons and
  neither of them the merge any more** — see the TASK-15700 section below.
  Prompts have no vector leg to hide any of this, so prompts are where it
  shows. **Do not read this cell as evidence that B2 did not land** — B2's
  reachability is pinned separately.
- **The `plain` 0.000s on `paraphrase` and `vocabulary_mismatch` are not free
  headroom either.** Those 22 queries were the honest target of the first P2c
  FEATURE candidate, and it was probed before it was built: PRF came back
  **NULL** — 0 of 22 rescued, 10 of 21 currently-hitting queries lost — under
  a pre-registered bar (TASK-15965). Read "The fourth retired P2c premise"
  under the admission protocol before proposing a query-widening fix for these
  cells; the probe machinery is reusable and the failure mechanisms are
  measured, not guessed.

**Hybrid's 0.152 of overall headroom is real but is mostly those two
classes.** Do not expect a fusion knob to reach it: `negation` needs a
capability the pipeline does not have, and the residual `prompt` misses have
now outlived the two levers that were supposed to reach them — the
cross-sub-leg merge was fixed and a widening that survives it did ship
(TASK-15700, 2026-08-13), and those four cells did not move. What is left on
that class is either a capability (stemming/expansion) or the FUSION
weighting that disqualified the constructions which answer all five. If a
future weighting change needs evidence that it *adds* something to the
classes already covered, it still needs a **new** vector-blind fixture
authored the way the existing one was (see the "VECTOR-BLIND KEYWORD
TARGET" sections in both fixture files).

**The bound on "nothing regressed", as it stood at the end of the weighting
arc.** It was a 49-document corpus in which every scored query except one
was already answered at rank 1, so there was very little left to damage —
**that half is what P2ab's fail-first authoring retired**: the corpus is now
172 documents and hybrid recall is 0.848 (0.826 as P2ab left it, before
TASK-15400's construction flip), so there is something left to
damage and something left to gain (see the headroom table above). The other
half stood on `k`: this harness measures at `k = 10`, while
the Library Search/RAG surface, at the time this was written, defaulted to a
hardcoded `LIBRARY_RAG_DEFAULT_TOP_K = 5` (half the fused candidate window,
since `_hybrid_search` fetches `top_k * hybrid_pool_multiplier` per leg). A
hybrid number here was therefore a statement about this corpus at k=10, not a
promise about a large library at k=5, and TASK-4110's live check answered the
k=5 half separately, against a real 64-document library through the running
TUI.

**That surface-is-tighter bound is retired (TASK-15020/B3).** The window's
depth is now the ACTIVE RAG PROFILE's `search.default_top_k` — 15 on the
shipped default profile (`hybrid_basic`), and configurable in Settings —
resolved through `library_rag_state.library_rag_profile_top_k`. The harness's
k=10 is no longer *deeper* than the surface it stands in for, so a hybrid
number here no longer needs the "but the real window only shows half of this"
caveat. It is still a statement about *this corpus*, and a profile tuned
below 10 (e.g. `fast_rag`, `long_context_rag` — both 5) puts a user back
under the harness's window: the profile, not a constant, is what to check.

Two mechanical effects behind the precision drops, so nobody re-derives
them later:

- **One passage per document.** Before TASK-3994 a document could occupy
  several top-k slots with several of its own chunks; fusing on document
  identity collapses those into one row showing the matched chunk. The
  freed slots fill with further documents, and since `precision@k` divides
  by `min(k, len(retrieved))` (see the `docs` column below), precision
  falls while recall/MRR/NDCG stay put. The trade-off is deliberate: one
  passage per document is the honest unit for an evidence list, and it
  costs a precision number that was partly counting duplicates.
- The **new golden query** is a single miss in a 16-query category, worth
  −0.062 recall on the keyword row for semantic and hybrid alike.

The discipline the table demonstrates: informational gated runs mid-arc,
**one** deliberate re-stamp at the end, and both sets of numbers in the PR.

### The third real re-stamp: P2ab (TASK-15020, 2026-08-11)

Eight tasks ran against these baselines without touching them. Every one of
those gated runs read `ENVIRONMENT_CHANGED` — first on `corpus_sha256` (the
corpus tripled in Task 3), then additionally on
`pipeline_config.source_types` (B2 added `prompts`; a **pre-existing key
whose value changed**, not a schema change). That was expected, recorded in
each task's report, and deliberately never "fixed" early. One re-stamp
closed it, and every delta in it belongs to one of five classes — no
surprises were stamped over:

| delta class | what moved | why |
|---|---|---|
| new category rows (`absent -> value`) | `negation`, `prompt`, `scoped` × 5 metrics × 3 modes | the fail-first classes exist for the first time |
| the scoped flip | hybrid `scoped` 0.000 -> **1.000** (MRR 0.163, NDCG 0.348) | B1 routing, measured — see the `scoped` bullet under Category meanings |
| semantic precision/F1 | `keyword` P 0.126 -> 0.117, `paraphrase` 0.103 -> 0.100, `vocabulary_mismatch` 0.106 -> 0.103 (+ their F1s) | **mechanical, not retrieval**: `precision@k` divides by `min(k, len(retrieved))`, and with 123 more documents semantic now fills all ten slots — `mean_docs_at_k` 9.105 -> 9.652, measured. Recall/MRR/NDCG on all three cells are byte-identical, which is what says retrieval did not move |
| hybrid `keyword` MRR/NDCG −0.001 | 0.945 -> 0.944, 0.957 -> 0.956 | the single relevant rank that moved in the whole old golden set: `kw-plant-maintenance-record`'s target 8 -> 9 (still rescued). `(1/8 − 1/9)/16 = 0.00087`, which is the whole delta |
| overall rows | semantic 0.974 -> 0.804, plain 0.355 -> 0.293, hybrid **1.000 -> 0.826** | the scored set grew 38 -> 46 with classes the pipeline fails |

**The old-queries deviation, as adjudicated ACCEPT in Task 3** — carried
here verbatim so the third and fourth rows above are never re-derived as an
unexplained regression: *"Full top-10 identity was measured, not achieved,
and it is not achievable when a corpus triples — no rewording fixes a tail.
What was held (and is the property the measurement rests on) is that every
old query's relevant ranks and every negative's behaviour are unchanged.
Nothing was reworded because nothing needed it."* Concretely: 45 of 135
before/after cells had identical top-10 lists and they are exactly the 45
`plain` cells; relevant-document ranks were unchanged on 44 of 45 queries in
every mode; `plain` still returns nothing for all 7 negatives.

### The fourth real re-stamp: TASK-15400 (2026-08-12)

The engine keyword leg's MATCH construction became configurable
(`SearchConfig.fts_match_construction`) and four candidates were swept
against this corpus at the shipped fusion parameters, under a decision rule
registered **before** the run: maximum leg-level census subject to three
hard constraints — (a) the vector-blind fixture keeps its hybrid rescue,
(b) no gated cell regresses more than 0.020 in any mode, (c)
`test_fts5_query_escaping.py` stays green with per-token quoting intact.

```
row              construction  census  resc  zero     P@k     R@k     MRR    NDCG  rescue  rank
and                       and      20     0    40   0.087   0.826   0.807   0.811     yes     9
and_trim    and_stopword_trim      21     1    39   0.089   0.848   0.809   0.817     yes     9
or                         or      28     9    11   0.089   0.848   0.751   0.774      NO     -
and_or            and_then_or      29     9    11   0.089   0.848   0.751   0.774      NO     -
```

**The winner was computed, not chosen: `and_stopword_trim`, census 21.**
The two high-census rows are disqualified, and — this is the arc's finding —
**for two different reasons, only one of which is about the match form**:

- under **`or`** the JOIN itself loses the vector-blind fixture: ten OR rows,
  target absent from the leg's top-10 and at leg rank 11 in the k=20 fusion
  window. No merge behaviour is implicated.
- under **`and_then_or`** the fixture's own sub-leg is *untouched* — its row
  is still an AND row, exactly as the design argued "by construction" — and
  the **MERGE** loses it anyway. `_keyword_search` merges its four sub-legs
  with `interleave_rankings`, a **round-robin**, so the media and
  conversations sub-legs falling back to OR injects rows that demote the
  untouched notes row from leg rank 1 to 2. Fusion consumes *leg* rank:
  `0.3/6 = 0.0500` (which strictly beats the vector rank-9 row's
  `0.049999999999999996`, by 6.94e-18) becomes `0.3/7 = 0.0429`, below the
  vector rank-11 row's `0.04375`, and the rescue is gone. The same
  displacement decomposes the scoped collapse **exactly**: the 4
  note-targeted scoped queries fall behind a media fallback row while the 3
  media-targeted ones keep rank 1 — 3/7 = 0.429, the measured cell to the
  digit.

So constraint (a) is a property of the **merge**, not of the construction.
That became **TASK-15700**, and its counterfactual margin was recorded here:
re-fusing the `and_then_or` pass with the fixture restored to leg rank 1 and
nothing else changed puts it back at **slot 10 of 10** — a merge fix rescues
it with zero headroom, so fixing the interleave is necessary and *not
sufficient* for a widening construction to ship.

**Both halves of that prediction held.** TASK-15700 (2026-08-13) fixed the
merge; `and_then_or`'s rescue came back **at exactly slot 10**, as forecast,
and the row was disqualified anyway — on constraint (b) this time, by
fusion rather than by the merge. The construction that did ship
(`and_then_prefix`) is a widening the fixed merge survives. See the
TASK-15700 section below.

**What the re-stamp absorbed: exactly ten of the 105 gated metrics, all up,
all hybrid.**

```
hybrid   category.prompt.f1          0.000 ->   0.036  (+0.036)
hybrid   category.prompt.mrr         0.000 ->   0.022  (+0.022)
hybrid   category.prompt.ndcg        0.000 ->   0.060  (+0.060)
hybrid   category.prompt.precision   0.000 ->   0.020  (+0.020)
hybrid   category.prompt.recall      0.000 ->   0.200  (+0.200)
hybrid   overall.f1                  0.157 ->   0.161  (+0.004)
hybrid   overall.mrr                 0.807 ->   0.809  (+0.002)
hybrid   overall.ndcg                0.811 ->   0.817  (+0.007)
hybrid   overall.precision           0.087 ->   0.089  (+0.002)
hybrid   overall.recall              0.826 ->   0.848  (+0.022)
```

The other 95 are `(+0.000)` — every `plain` and `semantic` cell, and every
hybrid category except `prompt`. Nothing regressed anywhere, and the
zero-movement of the two non-hybrid modes was **measured, not argued**: a
full three-mode run per construction produced byte-identical `plain` and
`semantic` cell dicts across all four rows.

**Which environment stamped this, and why it matters.** A baseline stamp
bakes the fingerprint, so the choice was made explicitly rather than by
running whatever was to hand. Two interpreters on the verification machine
disagree: the branch's own worktree venv carries `transformers 5.15.0 /
torch 2.13.0 / chromadb 1.5.9`, while the committed baselines were stamped
under `5.6.2 / 2.11.0 / 1.5.8`. **The stamp was run in the interpreter whose
fingerprint MATCHES the committed one**, with `PYTHONPATH` forced to this
worktree and the import provenance asserted inside the run, so the *code*
measured was the branch's and the *environment* recorded did not move. The
consequence is deliberate: this re-stamp changes metric values only — the
`environment` block is untouched, and the gate stays live for everyone on
the stack the baselines have always described. The newer stack is not
hidden: a gated run there still prints `ENVIRONMENT_CHANGED` (chromadb,
torch, transformers) and gates nothing. It also reproduced **all 105 cells
at (+0.000)** against these baselines, in both directions across the version
gap, which is the evidence that the gap is numerically inert here and that
the stamp is not concealing an environment shift inside a code change. If
you build a fresh worktree venv today (`uv pip install`) you will get the
newer stack and an informational gate; to get a *real* gate, use an
interpreter matching the fingerprint above — and assert
`tldw_chatbook.__file__` inside the run, because an editable install can
resolve to a different checkout entirely.

**The negative-composition trade-off was NOT observed, and it could not have
been.** The sweep counted OR-form rows arriving on the 7 negative queries,
expecting widening to buy recall by admitting junk. Every row of the matrix
reports zero — but a direct probe of the denominator shows all seven
negatives return **zero keyword-leg rows under every construction, OR forms
included**: their vocabulary is absent from the corpus outright. The zero is
a property of this golden set's negatives, not evidence that OR widening is
quiet. **Never quote it as a clean bill of health for widening**; the
instrument is not vacuous in general (the same counter found an OR-form row
at hybrid slot 10 for a non-negative query), it is vacuous *here*.

**One fixture changed role rather than content.** `pm-vendor-chaser` was
admitted 2026-08-10 as a fail-first prompt query and is now answered at
hybrid rank 9 — so the admission protocol below would **reject it as a new
candidate today**. It is deliberately kept as a *retained positive*: it is
the only golden query whose cell measures the shipped construction, and
re-authoring it fail-first would delete the arc's own evidence while
returning the cell to 0.000. Its dated `# admitted:` receipt is left intact
(it is true of the day it was written) and a `# retained:` line beside it
records the conversion. That comment is the only fixture edit in this arc —
and because `corpus_sha256` hashes the fixture files' **bytes**, a comment
is a fingerprint change: the digest moved, the 105 metrics did not (verified
by diffing the two stamps), and the committed baselines match the committed
fixtures.

### The fifth arc, and the re-stamp that did NOT happen: TASK-15700 (2026-08-13)

TASK-15400 closed by naming its own blocker: the keyword leg merged its four
source sub-legs with `interleave_rankings`, a **round-robin over sub-leg
position**, so one sub-leg's row COUNT decided every other sub-leg's leg
rank — and fusion consumes leg rank. TASK-15700 fixed that merge and then
**re-ran the construction sweep under it**, with two new rows (`prefix`,
`and_then_prefix`) pre-registered before the run.

Two things shipped. **Part A**: the sub-leg merge is now form-tiered —
primary-form rows (tier 1) wholly precede fallback-form rows (tier 2),
interleaved within each tier — so a sub-leg that found nothing can no longer
demote a sub-leg that found something. **Part B**: the engine leg's default
MATCH construction moved `and_stopword_trim` → **`and_then_prefix`** (full
AND first; per-token prefix matching only for a sub-leg whose AND returned
zero rows).

**The six-row re-run, gated, hybrid, k=10, 60 golden queries:**

```
row              construction  census  resc  lost  zero  rescue  rank   gate (105 cells)
and                       and      20     0     0    40     yes     9   REGRESSED (5 past 0.02)
and_trim    and_stopword_trim      21     1     0    39     yes     9   PASSED  (0 moved)
or                         or      28     9     1    11      NO     -   REGRESSED (12 past 0.02)
and_or            and_then_or      29     9     0    11     yes    10   REGRESSED (8 past 0.02, 5 past 0.05)
prefix                 prefix      23     3     0    36     yes     9   PASSED  (0 moved)
and_pfx       and_then_prefix      23     3     0    36     yes     9   PASSED  (0 moved)
```

`lost` = census hits the CONTROL row answered and this row no longer does;
`resc` = the control's zero-row queries this row's leg now answers. Hard
constraint (b) was measured in **all three modes** by the gate's own
`compare_or_update` against the committed baselines, not by the sweep:
`plain` and `semantic` moved zero cells under every construction.

**What the merge fix bought, measured.** `and_then_or` — the row 15400
disqualified on the vector-blind rescue — moved **rescue `NO` → `yes` at
slot 10, and scoped recall 0.429 → 1.000**, and the leg's own form stamps
say why: `['and', 'or', 'or', …]`, the untouched AND primary leading its nine
tier-2 fallbacks instead of being round-robined behind them. That is Part A's
payoff and it reproduces 15400's decomposition in reverse (the four
note-targeted scoped queries come back; the three media-targeted ones never
left).

**And it still does not ship.** `and_then_or` is now disqualified on
constraint **(b)** instead: 8 gated cells past 0.02, 5 past the 0.05 fail
band (paraphrase and vocabulary_mismatch mrr/ndcg, `overall.mrr` −0.056),
with recall on those categories at +0.000 — the signature of correct rank-1
answers being re-ranked rather than lost. The mechanism, recorded so a
future arc need not re-derive it: **tier 2 confines fallback rows inside the
keyword leg, but tier 2 still enters FUSION**, where a fallback row that also
carries a vector rank becomes a MERGED row outscoring any fts-only row.
Sharper still — all five regressing queries have an EMPTY keyword leg under
the shipped default, so under `and_then_or` every sub-leg falls back, the leg
is **100% tier 2**, tier 1 is empty and the partition is the identity
function. **Part A is structurally INERT on exactly the queries that
disqualify the row**; no tiering change could have saved it. Its 8-census
lead over the winner is a fusion-weighting question, not a merge one.

`or` fails for a different reason again, and the merge was never its cause:
the fixture is **absent from the leg's top-10 entirely**, displaced inside
the notes sub-leg's own bm25-ordered, LIMITED result set before any merge is
consulted. A widening PRIMARY puts every row in tier 1, so tiering has
nothing to tier there.

#### The decision record — the rule's winner and the owner's ruling, never conflated

1. **The pre-registered rule (15400's, verbatim) was applied and ran to
   completion.** The census-maximal row `and_then_or` (29) is disqualified on
   (b); `or` on (a) and (b). Qualifiers: {`and_stopword_trim` 21, `prefix`
   23, `and_then_prefix` 23}. **Max census 23 is a TIE**, and the two tied
   rows are measurement-identical on every captured axis — same census
   hit-set, all 105 gated cells unmoved, **all 60 per-query hybrid top-10s
   and all 60 keyword-leg top-10s identical**, `lost` 0 both ways. The rule's
   first tie-break is fewest extra FTS **statements**, measured over the 60
   golden queries: **240 vs 460**. **The rule's winner is `prefix`.**
2. **The owner ruled `and_then_prefix` ships instead**, applying the standing
   stability-over-quick-wins ruling to a dimension the tie-break predates:
   `prefix` widens as a PRIMARY, so Part A's tiering protects it not at all
   and it carries a measured self-displacement failure mode (12
   prefix-competitor docs + 1 exact-match doc, top_k=5 → the exact doc gone,
   inside ONE sub-leg where tiering has nothing to tier). `and_then_prefix` is
   immune to that by construction — a non-empty AND primary is never widened.

**Read the shipped value correctly: `and_then_prefix` is NOT the rule's
output.** The rule produced `prefix`; the tie between two
measurement-identical qualifiers was decided by a standing owner ruling. Both
halves are stated at every site that records the outcome (`config.py`,
`_escape_fts5_query`, the flipped test docstrings), and the shipped-default
pin is named `test_the_shipped_default_is_the_owner_ruled_construction` for
exactly that reason.

**The price, stated with its bound.** 220 extra SQLite statements over the 60
golden queries — **92% of sub-legs actually fall back**, not the "up to" a
definition count implies. That 92% is an **UPPER BOUND belonging to this
172-document corpus**, not a forecast: the fallback fires precisely when the AND
primary finds nothing, so a denser corpus hits the primary more often and the
count falls. Wall time is indistinguishable at this scale (~1.0–1.5s per
row), so the tie-break was decided on work performed, not time observed.

#### What the flip bought, and what it is not

Leg-level census **20 → 23 of 53** against the pre-arc control (21 → 23
against the outgoing default), residual zero-row **40 → 36 of 60**. The three
ids, named because a bare `23` survives trading one hit for another:
`pm-vendor-chaser` (also reachable by the outgoing trim), plus the two the
prefix fallback alone reaches — `kw-quillon-mast` ("guy **tension**" against a
document that says "guy **tensions**") and `kw-thimble-relay` ("relay board
**swap**" against "**swapping**/**swapped**"). That is the shipped class in
one line: **a typed content word now matches a document word that STARTS with
it.** The reverse does not hold — typing `tensions` still will not find
`tension` — so this is prefix widening, not stemming.

Residual zero-row after the flip, **re-measured per category** (leg-level,
negatives included, 2026-08-13, at the shipped default):

| category | zero-row / n |
|---|---|
| `keyword` | 1 / 16 |
| `negation` | 2 / 3 |
| `negative` | 7 / 7 |
| `paraphrase` | 13 / 13 |
| `prompt` | 4 / 5 |
| `scoped` | 0 / 7 |
| `vocabulary_mismatch` | 9 / 9 |
| **total** | **36 / 60** |

with the census at 23/53: `keyword` 15/16, `scoped` 7/7, `prompt` 1/5,
`negation` 0/3, `paraphrase` 0/13, `vocabulary_mismatch` 0/9.

**The new default is NOT a superset of the outgoing one by construction.**
`and_then_prefix`'s primary is the FULL AND — every token, function words
included — so a sub-leg whose full AND returns rows never seeks the
trim-only hits. That nothing was lost (`lost` 0 against both the control and
the outgoing default) is a **MEASURED fact about this corpus**, never a
structural guarantee.

#### The vector-blind fixture is on the boundary, not above it

Under the shipped construction `kw-plant-maintenance-record`'s target
(`note-saltmarsh-hide`) holds **slot 9 of 10** — exactly the slot the
outgoing default gave it. **This is not headroom, and the distance to slot 11
must not be quoted as margin.** The row immediately below it is a
**MATHEMATICAL TIE**: the fixture's fts-only score is `(1−α)/(rrf_k+1) =
0.3/6` and that row's is `α/(rrf_k+9) = 0.7/14`, and `0.3/6 == 0.7/14 == 1/20`
**exactly** in rational arithmetic (4 ULPs apart in IEEE-754). It keeps its
slot on `reciprocal_rank_fusion`'s documented `(-score, fts_rank,
vector_rank)` tie-break, **not on any margin** — and it sat on that same
boundary under the old default too. Nothing in this arc adds margin and the
shipped construction spends none. What would displace it is a **MERGED
(fts+vector) row**, which is exactly what a widening PRIMARY manufactures:
`and_then_or` inserts one at slot 9 and pushes the fixture to slot 10. Fusion
parameters (`rrf_k`/`alpha`/`pool`) own that margin and stayed out of scope.

#### AC#6: the re-stamp is a disclosed NON-EVENT

**Zero of the 105 gated cells moved.** Stronger than a count: the gate's own
105 printed cell lines are **byte-identical** between a run at the old
default and a run at the new one, in all three modes. So no baseline file was
touched in this arc, no stamp was manufactured, and the gate's environment
fingerprint is the one TASK-15400 committed. A construction change that
buys leg-level coverage the vector leg was already covering is *supposed* to
be invisible here — the gain shows in the census and in the zero-row count,
which is why those two numbers exist.

One deliberate omission for the same reason: **the fixture files were not
edited.** `corpus_sha256` hashes their bytes, so even a comment correcting
15400's dated receipts (`golden.toml`'s `# retained:` line still names
`and_stopword_trim`) would move the fingerprint and force a re-stamp this arc
did not earn. Those receipts are true of the day they were written; this
section is the current state.

### The sixth arc, and the second re-stamp that did NOT happen: TASK-16071 (2026-08-14)

The Library's **plain four-seam keyword path** (notes, media, conversations,
prompts — each its own query under its own `limit=top_k`) used to merge its
seams by **concatenating them in a fixed source order**. Every row carries
`score=None` on purpose, so nothing sorted the concatenation: a row's
cross-seam position was a function of its SOURCE TYPE and of how many rows the
earlier seams happened to return, never of how well it matched. Any pass
matching `top_k` notes buried every media, conversation and prompt hit behind
them, and every downstream cut cut exactly there.

It now merges with **`interleave_rankings`** — the engine's own rank-fair
primitive, not a re-implementation — keyed on `(provenance.source_type,
source_id)`: each seam's rank-1 row, then each seam's rank-2 row, and so on,
with `_KNOWN_KEYWORD_SOURCE_TYPES` order breaking ties *within* a position
(a pinned convention, not a relevance claim). **ORDER only — no truncation
added**, since a four-seam query has always been able to return up to
`4 × top_k` rows and the cut belongs to the consumers. **No tiering, deliberately:**
every seam builds its MATCH through `build_fts_match_query` and nothing else,
so all four rankings are all-primary and TASK-15700's tier design has nothing
to tier here; it applies the day this path gains fallback forms. The rule, the
worked examples and both of those decisions are written at the merge site in
`Library/library_local_rag_search_service.py`, which is where the next reader
will be standing.

**The gate saw nothing, and that was predicted before the change landed.**
All **105 of 105** gated cells read `(+0.000)`, and stronger than the 3dp
print: re-compared with float equality, **0 of 105 metrics differ bit-exactly**.
No baseline file was touched. The reason is a census, not luck — **every one
of the 60 golden queries returns ≤1 row under the shipped plain pass** (rows>1:
0; spanning more than one seam: 0), measured at k=10 *and* at k=200. An
order-only change over a list of length ≤1 is the identity function, so on
this instrument a moved cell would have been a **STOP**, not a success. That
prediction was registered before the merge changed, and it is the second arc
running whose re-stamp is a disclosed **non-event**.

**So the proof lives in two places outside the golden cells**, both of which
exist because the instrument structurally cannot see this fix: five always-on
pins on the real path with four real databases
(`Tests/Library/test_library_keyword_cross_seam.py` — displacement,
rank-fairness, single-seam byte-identity, no-truncation, prompts-seam
participation; **RED 3/5 on the old concatenation**), and the PRF probe's
oracle control, re-run under both merges in one session.

**Plain keyword recall does not move, and the reason is worth keeping.** It is
**0.84375 = 13.5/16** — thirteen queries at 1.000, one at 0.500, two at 0.000
(the `precision` cell, 0.875 = 14/16, hides the half-miss). All three lost
targets are **genuine non-matches**, absent from the merged list even at
DEEP_K=200 where the whole pass returns 0-1 rows: a conjunctive MATCH with
plural-only widening against corpus text that inflects differently
(`slipping`/"slips", `swap`/"swapped", `rollback`/"rolled back"). **Zero
seam-burial.** The lever on that cell is the term construction (TASK-3997's
question, TASK-15400's territory), not the merge — worth stating because a
merge fix is the intuitive place to look for it and it is the wrong place.

**What the control did show: one number was two defects.** Under the oracle
feed (below, "the fourth retired P2c premise"), the pre-registered TF-8
selector goes **8/22 → 14/22** and the rarest-DF selector **15/22 → 19/22**,
and the ENTIRE gain is the conversation column (0/6 → 6/6 and 2/6 → 6/6) while
**media moves +0 under both selectors** (1/9 and 6/9, unchanged). Conversation
burial was seam **ORDER** and it is now gone completely; the media shortfall is
per-seam **VOLUME** — those targets sit ≥7 deep inside the media seam against a
reachable depth of 3 in a ten-row window, so no ordering change reaches them.
The honest observability bound recalibrates **≥15 of 22 → ≥19 of 22**.

**The residual headroom is therefore the media column, and it is not an
ordering problem.** The levers that could move it are the per-seam budget (pull
deeper per seam before any cross-seam cut) or the widening construction; a
further ordering change is not one of them, and no future arc should spend
itself there expecting the media column to answer.

**The change is not free, and both costs are measured.** In the PRF probe's
widened-feed regime the collateral loss count is identical either side (10 of
21 hitters) but **two of the ten are different queries** — the rotation pulls
two conversation targets back inside k and pushes two NOTE targets out, so the
displacement cost lands on the notes seam. And the fed top-M window went **18%
→ 54% label-only**: the merge changes *what* a top-M consumer sees, not merely
the order. In the product, Search mode's evidence list now interleaves by rank
across sources for any query matching more than one source — a visible change,
disclosed in `Docs/User_Guide/library/search-and-rag.md` and live-checked.

**What this arc did NOT touch, stated so nobody re-measures it.** The engine's
hybrid keyword leg is a different module: TASK-15700's leg-level census
(23/53) and residual zero-row count (36/60) are properties of `RAGService`'s
sub-leg merge and its MATCH construction, and nothing in the retrieval path
imports `library_local_rag_search_service`. Those two numbers were therefore
not re-measured and cannot have moved; the observable consequence that *was*
measured is that hybrid's gated cells are bit-exact unchanged.

**The production consumer with a hard cut, named (final review, 2026-08-14).**
Enumerating consumers of the private `_search_keyword` was one level too low:
the public `search()` has a second production caller —
`Agents/library_rag_tool_provider.py:216-219` issues `mode="rag"` and `:250-252`
truncates to `_MAX_TOP_K` (10). It is the DEFAULT Console Library retrieval
tool (bound whenever `direct_library_tools` is off) over notes/media/
conversations. Under a plain profile this is the one place in production where
**set membership**, not merely order, changes: the pre-fix window could be ten
note rows carrying real `content`; the post-fix window is a ~4/3/3 rotation
whose media and conversation rows carry `"Matched media · {type}"` /
`"Matched conversation · N messages"` and no document text. That is the same
mechanism as the probe's 18% → 54% label-only inversion, landing on an LLM's
evidence rather than a probe's feed — an anticipated cost of rank-fairness on a
path whose rows are not all text-bearing, and the sharpest argument for
TASK-16174's expansion/fetch work. Not a regression of this arc's ACs (order
was the change; the label-only rows were always in the corpus), but it belongs
in the record rather than only in a probe's price line.

## Fail-first authoring: the admission protocol

A fixture that the pipeline already answers measures nothing. P2ab
therefore **probed every candidate against the real stack before admitting
it**, and the protocol is now the rule for adding any query to this set:

1. Write the candidate and its target document.
2. Run it through the real seam at `k=10` in **all three modes**.
3. **Admit it only if the target misses the top-10 in both vector-bearing
   modes** (hybrid *and* semantic), each having actually run and returned
   rows. The `plain` rank is recorded, never required.
4. Write the measurement into the fixture as an `# admitted: <date>
   hybrid=.. semantic=.. plain=..` receipt (`fixture_probe.admission_comment`
   produces it; an always-on test checks both its presence and its shape).
5. If the candidate passes today, **it is not admitted** — and the rejection
   is recorded with its ranks, because a class that cannot be made to fail
   is itself a finding.

31 candidates were probed this way and **15 admitted**. Per-class outcomes,
which are the recorded evidence P2c should plan against:

| class | authored | ADMITTED | outcome |
|---|---|---|---|
| `scoped` | 8 | **7** | admitted by *routing* — `plain` rank 1 on all seven, vector-blind on all seven. B1 flipped hybrid 0.000 -> 1.000 |
| `negation` | 6 | **3** | the admitted three never name the negated aspect; the rejected three (ranks 4, 6, 7) have exception documents that *are* about the queried aspect, so topical proximity answers them before negation-blindness can matter |
| `prompt` | 5 | **5** | structural: no vector index by construction. Probed anyway, so the record shows absence rather than assumption |
| `acronym` | 6 | **0** | **UNFAILABLE on this corpus and model** |
| `compositional` | 6 | **0** | **UNFAILABLE on this corpus and model** |
| `precision_pressure` | 0 | 0 | not authored, deliberately |

**The two unfailable classes are P2c evidence, and the strongest kind — a
measured negative.**

- **`acronym`.** Bare acronym in the query, expansion spelled out in the
  target, letters absent from it, plus the reverse direction and decoys
  using the letters in another sense. `all-MiniLM-L6-v2` bridges MTBF / PPE
  / BOM / RTO / UPS to their expansions unaided — ranks 1, 1, 1, 2, 2. The
  one anchor deliberately diluted (four extra documents so the topical
  anchor could not resolve the query alone) moved its target from rank 1 to
  rank 4 and no further. **This retires acronym expansion as a P2c feature
  premise**, the same way P1's vocabulary-mismatch collapse retired query
  expansion.
- **`compositional`.** Answer documents holding both halves of a stated
  conjunction in different sentences, with decoys holding each half
  squarely. Ranks 1, 1, 1, 1, 2, 2; anchor dilution did not move it. A
  document containing both halves is simply the nearest neighbour of a
  query naming both. **Retired as a P2c premise.**
- **`precision_pressure` was not authored, deliberately.** The class needs
  many near-relevant decoys *and* a crisp relevance boundary between them,
  and the two requirements pull against each other: with a dozen
  near-duplicates the cell measures where the label was drawn, not what
  retrieval did.

All 16 rejected candidates' query texts, targets and ranks are preserved in
`golden.toml`'s measured-outcomes block, so both unfailable rulings can be
audited in place rather than by re-authoring the candidates.

**Two traps this protocol caught, worth knowing before authoring more:**

- **Scope size is the `scoped` class's lever, and it was measured.** With 32
  documents in scope, 1 of 8 candidates failed; at 80, six; at 100, seven.
  A top-10 over a small scope returns nearly everything in it, so the cell
  would report recall 1.000 for a reason unrelated to retrieval. The shipped
  scope is pinned at **exactly 100** slugs, shared byte-identically by all
  seven fixtures and digest-pinned (`SHIPPED_SCOPE_SHA256`) so a
  coordinated swap at the same size still fails a test. Trimming it would
  have raised the before-number from 0.000 to roughly 0.43 with every test
  green.
- **A candidate whose subject is the corpus's only document on that subject
  measures corpus sparseness, not the pipeline.** Every class needed anchor
  company before its ranks meant anything; 24 documents were added purely
  for that.

### The fourth retired P2c premise: PRF, probed before built (TASK-15965, 2026-08-13)

> **Every probe figure in this section was re-measured on 2026-08-14 under the
> rank-fair four-seam merge (TASK-16071, the arc above), and the ones that
> moved carry their pre-16071 value inline as dated history.** The section was
> first written under the path's fixed-order concatenation; that merge no
> longer exists, so a figure quoted without its date would describe a path the
> code does not have. **The verdict did not move: NULL, and PRF stays
> retired.** It is now measured on a wider basis — nine grid points instead of
> one — and against a higher observability ceiling (14 of 22 cells rather than
> 8 under the pre-registered selector), so the null is stronger than when it
> was recorded, not weaker. What did not change at all: not one of the 105
> gated golden-set metrics moved (bit-exact), because the shipped plain pass
> returns ≤1 row per golden query and an ordering change over a one-row list
> is the identity.

The three premises retired above died to things the instrument could already
show: query expansion to a **cell** (`vocabulary_mismatch` reads 1.000 in both
vector modes — there is no ceiling left to improve against), `acronym` and
`compositional` to **fixture** probes (31 candidates authored, 12 of them in
those two classes, 0 admitted because today's pipeline answers them all). The
fourth is the first **feature** candidate to die the same way.
**Pseudo-relevance feedback** was next in the approved P2c cost order; it was
probed and priced on this corpus **before any production code was written**,
against a bar pre-registered in
`Docs/superpowers/specs/2026-08-13-rag-p2c-prf-fail-first-design.md`.
**The verdict is NULL. No PRF code exists, and the arc ended there.**

**The bar, its four clauses verbatim, against the measurement:**

| # | pre-registered clause (spec, verbatim) | measured | result |
|---|---|---|---|
| 1 | "≥5 of the 22 plain-failing queries reach their target in the second pass's top-10" | **2 / 22 rescued** at the base point (**0 / 22** pre-16071), and **2 / 22** is also the maximum over all nine grid points | **FAIL** |
| 2 | "zero currently-hitting plain queries (any category) lose their target" | **10 of 21 hitters lost**, every one at rank 1 today (10 pre-16071 as well — but two of the ten are *different queries*; see the loss diagnosis) | **FAIL** |
| 3 | "zero new rows on negatives" | 0 new rows and 0 new documents on all 7 — but **structural**, see below | PASS, worth nothing |
| 4 | "the negation guard: no negation query's row set grows with assertion-side junk (measured, reported; expected to bind)" | all 3 negation queries went **0 → 30 rows, +10 new documents** | BINDS, as pre-registered |

Two gating clauses fail. Clause 1 misses its bar by 3 of the 5 rescues it
needed — pre-16071 it missed by all 5, and the narrowing is the merge's doing,
not PRF's (both rescues are conversation targets, the seam TASK-16071 freed).
Clause 4 was never a gate; clause 3's pass is not evidence of safety (below).

**Step 0 — fireability, which decided the regime in one command.** Before any
grid point, the probe asked whether PRF can fire at all: does the shipped
first pass return ANY rows for the 22 plain-failing queries? **0 / 22 on the
shipped four-seam AND-strict pass** — the builder requires every query term to
appear together in one document, and no document in the corpus holds all of a
paraphrase query's content terms (a corpus-wide property of the AND across
query terms — NOT a fact about the target's vocabulary; PRF feeds on any
returned rows, not the target's), so PRF's classic mechanism has literally
nothing to feed on. That alone retires the un-varianted premise. The spec's
ONE licensed variant then activated (an OR-of-content-terms pass used **for
feedback selection only**, never shown to a user): **18 / 22 fire** under it.
Every before-column in every table below remains the SHIPPED pass. (This
census is **byte-identical either side of TASK-16071** — an ordering change
cannot create a row, which is the same identity the golden cells prove.)

**The base point (N=8 terms, M=5 fed documents; RM3 `tf/|D|`, the
pre-registered derivation): 2 / 22 rescued, 10 of 21 hitters lost.** Both
rescues are conversation targets — `pr-platform-offline` →
`conv-platform-offline` and `vm-nearsightedness` → `conv-myopia-visit`, the
seam the rank-fair merge freed. **Pre-16071 this point read 0 / 22**, and that
zero kept the grid shut: the {4,8,16}×{3,5,10} sweep is licensed only if the
base point shows signal ("a null at every base point is a null, not an
invitation to search the grid until something moves"), so **one point ran and
one point was recorded**. With the base point off zero the sweep opened for
the first time and **all nine points ran**: the maximum rescued anywhere on
the grid is **2 / 22 against a bar of ≥5**, and clause [2] fails at 9-10
hitters lost at every single point. State the direction carefully — the *gap
to the bar* narrowed (0 → 2 against ≥5); what widened is the **evidential
basis**, a nine-point sweep saying what one point used to.

The 10 losses are diagnosed, not just counted: re-running the *same* expression
at k=200 separates "the expansion never reached the document" from "it reached
it and lost its slot". **0 of the 10 are unmatched — 9 seam-displaced, 1
merge-displaced** (pre-16071: 8 and 2).

**The count is the same either side of the merge change; the identities are
not, and that swap is a disclosed COST of the rank-fair merge.** Two of the
ten differ: pre-fix it lost `kw-ashgrove-pump` and `kw-drayton-conveyor` —
conversation targets merge-displaced at position 21, which the rotation now
pulls back inside k — and post-fix it loses `kw-plant-maintenance-record` and
`sc-meter-box-key`, both **NOTE** targets. The direction is the rotation's
price: media and conversation rows now take top-M slots a full notes seam used
to monopolise, so in a widened-feed regime the displacement cost lands on the
notes seam. Net-neutral on this clause and on the verdict — named here because
"10 lost both ways" would have been a misleadingly quiet way to report it.

Under the shipped
rank-fair three-seam rotation a within-seam rank `r` row lands at merged
position at most `3r-1`, so a merged-position-`p` row has `r ≥ (p+1)/3` — the
shallowest post-fix media miss sits at within-seam rank ≥7 against a reachable
depth of 3 at k=10; no ORDERING change can surface it (the bound is rank-fair-
qualified: a media-first concatenation could, but is not on the table). The
loss channel is **pure dilution**:
expansion-term rows
evicting a rank-1 target from a 10-row per-seam budget. (This refuted the
implementer's own prior, which was that the probe's expression — the engine's
content-token form, without `build_fts_match_query`'s plural/singular widening
— would strand documents outright.) On the *target* side the dominant mechanism
is the opposite one: 10 of the 18 fired queries are UNMATCHED at k=200 — the
derived terms never touch the target document. That is the "poison feed" the
spec pre-registered as PRF's central risk, measured.

**The axis control: a narrower selector rescues one, not five.** Outside the
pre-registration (so it can never ADMIT), the same real feed was re-derived
with a rarest-by-corpus-DF selector: **1 / 22 rescued at N=8 and 1 / 22 at
N=4** (**0 / 22 at both** pre-16071), the single rescue being
`pr-platform-offline` — a conversation target, the same mechanism as the base
point's two. Collateral damage stays 3 losses rather than 10, **unchanged in
count AND identity** across the merge change (`kw-quillon-mast`,
`kw-verdigris-coating`, `sc-storm-overflow-record`, all three merge-displaced
rather than evicted from their own seam). We looked down the TF-vs-DF axis; it
is one rescue against a bar of five.

**How many cells a rescue could have been seen in — the corrected framing.**
The probe's rescue-channel control feeds PRF the target document *itself* (the
best expansion any feedback set could produce). Read this before the floor —
and read the columns as pre-16071 → **as shipped**, because the merge change
is the only difference between them (same feed, same path, same k, same
corpus, same selectors, re-measured in one session by reverting the merge site
and restoring it):

| selector (oracle feed, same path, same k) | N | reaches top-10 | note | media | conversation |
|---|---|---|---|---|---|
| TF `tf/\|D\|` — the pre-registered derivation | 8 | 8 / 22 → **14 / 22** | 7/7 → 7/7 | 1/9 → 1/9 | 0/6 → **6/6** |
| rarest-by-corpus-DF — *ranking key only* | 8 | 15 / 22 → **19 / 22** | 7/7 → 7/7 | 6/9 → 6/9 | 2/6 → **6/6** |
| rarest-1, query side dropped — *illustration, changes two things* | 1 | 22 / 22 → 22 / 22 | 7/7 | 9/9 | 6/6 |

**22/22 of the oracle expressions match their target at k=200 in every row**,
so every miss is displacement, not a control that failed to reach its document.

**The whole of the gain is the conversation column, and the media column did
not move at all** — which splits what used to read as one defect into two. The
conversation targets were buried by seam ORDER: their within-seam rank was 1
all along, and all six now read merged position **3**, the conversation seam's
rank-1 slot in the harness's three-seam rotation. The media misses are buried
by per-seam VOLUME: the eight TF-8 misses sit at deep merged positions 18-82,
i.e. within-media rank ≥7 by the `r ≥ (p+1)/3` bound above, against a
reachable depth of 3 in a ten-row window — **no ordering change reaches them,
and none ever could.** The note column is a floor rather than a risk: notes
break the within-position tie, so a notes rank-1 row holds merged position 1
by construction, and all 7 read position 1 in both selector rows before and
after (read off the printed detail table, not inferred).

The defensible reading, restated for the shipped path: the plain four-seam
path merges **rank-fairly** (TASK-16071 — seam order decides nothing but ties),
and what still binds is the **per-seam `top_k`**: under the harness's
three-seam plain fan-out a seam can reach only ~⌈k/3⌉ of the k merged slots,
so a target deeper than that *inside its own seam* is still unreachable — and
**how hard that bites depends on expansion BREADTH, which is the selector's
property, not the path's.** So the honest bound on this null is **≥19 of 22
cells observable, PRF rescued 2 at its base point** (pre-16071: ≥15
observable, 0 rescued).

An earlier version of this probe printed the stronger claim — an 8/22 "ceiling
imposed by the four-seam path", "14 of 22 never observable" — and its review
refuted it by re-running the control with only the ranking key swapped. **Do
not restate the ceiling form**: it is an artefact of one term selector, and the
correction makes the null *stronger*, not weaker. The path property that
survives was filed on its own as **TASK-16071** (it was never about PRF; it
prices any query-widening technique on this path) and **shipped 2026-08-14** —
the rank-fair merge above. What it did *not* fix is the per-seam volume
constraint, which is where the residual headroom now sits; see "The sixth arc"
above.

**Two spec expectations the run corrected, recorded rather than smoothed:**

- **The negatives guard never becomes live.** The spec expected clause 3 to
  become "a REAL guard under the OR-feedback variant". It does not: a negative
  query's content words are absent from the corpus, so the OR feed returns 0
  rows too, no terms are derived and no second pass runs. All 7 read 0 → 0 in
  both regimes. **A guard that cannot bind is a property of the fixture class,
  not evidence of safety.** The live junk evidence is clause 4's negation row.
- **The price is real and was paid.** 211 content fetches per grid point over
  60 queries — one read per fed row, because four-seam media and conversation
  rows carry no document text (`"Matched media · {type}"`, `"Matched
  conversation · N messages"`), so without the fetch the feed would have
  skewed silently toward notes. **113 of 211 fed rows (54%) are label-only**
  — pre-16071 it was **39 of 211 (18%)**. The fetch count is identical either
  way (one read per fed row); what changed is the composition of the top-M
  window, and label-only went from the exception to the dominant case, because
  the rank-fair rotation puts media and conversation rows into slots a full
  notes seam used to hold. **The merge changes WHAT a top-M consumer sees, not
  only the order it sees it in** — a live consequence for anything downstream
  that reads a top-M window (RAG Answer's evidence, a PRF-style feed, a future
  re-ranker) and assumes a fed row is self-describing.

**One more finding for any future term-derivation candidate on this corpus:**
the engine's `_FTS5_STOPWORDS` (67 words) is too short for TF-based derivation
here — `rather`, `once`, `each`, `taken`, `through`, `back`, `same`, `before`
survive into the expansion lists and do the expanding.

**Where the machinery is, for the next candidate.** The probe is reusable; the
next candidate should start from it rather than re-invent it:

- `harness/prf_probe.py` — pure, pinned functions: `derive_expansion_terms`
  (RM3 `tf/|D|` as exact `Fraction`s — floats re-admit doc-order dependence
  through the ranking key on a corpus whose documents run 39→889 words),
  `compose_prf_expression`, `compose_feedback_expression`, `ProbeQueryResult`.
  It **imports** the engine's `_FTS5_STOPWORDS` / `_quote_fts5_token` /
  `_fts5_query_tokens` rather than re-implementing them.
- `test_prf_probe.py` — 28 always-on pins (no gate, no model), including the
  tokenizer-equivalence pin against `RAGService._fts5_term_key`.
- `test_prf_probe_run.py` — the gated run and the idiom worth copying:
  **fireability census FIRST**, then oracle/observability controls, then the
  grid, then guards derived from a fresh baseline pass at probe time (never
  hardcoded), then a verdict computed clause-by-clause with an assertion that
  a non-pre-registered selector can never reach it. Re-run with
  `RAG_EVAL=1 .venv/bin/python -m pytest Tests/RAG_Eval/test_prf_probe_run.py -s -q`.

The committed record of the run is **TASK-15965**'s Implementation Notes (the
verdict table, the corrected observability, both controls); the arc's full
report with the probe's verbatim output is
`.superpowers/sdd/2026-08-13-rag-p2c-prf-fail-first/task-2-report.md`, which is
an untracked SDD working record — re-run the gated module above if you need
the numbers and do not have that directory. The next candidate in the cost
order (a clarification gate) is filed as **TASK-16072** with a pointer to this
machinery.

## Fingerprint semantics: "environment changed" is not a regression

Every baseline carries an environment fingerprint over the load-bearing
embedding stack (TASK-3998): the embedding model string, the installed
`transformers`, `torch` and `chromadb` versions — the packages the
harness's real embedding/retrieval path (`Embeddings_Lib._HFEmbedder` ->
`transformers.AutoModel` + `torch`, with `chromadb` doing ANN retrieval)
actually loads — a SHA-256 over both fixture files (length-delimited, so
moving a byte between them cannot produce the same digest), and
`sys.platform`. `sentence-transformers` is recorded too, but only in
non-compared informational metadata: it is not on this harness's load
path, so a version bump there must not force a re-baseline the numbers
never asked for. A `pipeline_config` block (`k`, profile name,
`source_types`) is compared alongside the fingerprint, prefixed
`pipeline_config.` in any diff, so a `k` change reads distinctly from an
embedding-environment change.

When the current run's fingerprint (or pipeline config) does not match the
committed baseline's, the gate reports `ENVIRONMENT_CHANGED` and **does not
score the run at all** — it is not a pass because retrieval held up, it is a
pass because the numbers were never comparable in the first place. This
matters concretely: these baselines were stamped on `darwin`. On a
different platform or with different transformers/torch/chromadb versions,
the gate will go green having checked *nothing* — it prints a `NOTE:`
naming the differing keys and asking you to re-stamp on that machine. Read
that note; do not read the green exit code alone as "retrieval is fine
here."

A missing baseline is treated as a **failure**, not a pass — the same
reasoning as pytest's own "no tests ran": a green gate that checked nothing
is more dangerous than a red one.

## Category meanings

Every golden query carries one of the seven categories in
`goldenset.CATEGORIES`. A category must not be declared ahead of its
fixtures (`test_every_declared_category_is_required_and_populated` pins
that in both directions), which is why the two unfailable classes above are
absent from the tuple entirely rather than sitting in it empty:

- **`keyword`** — the query shares literal tokens with its relevant
  document(s). Exercises the FTS/keyword leg (plain's four-seam path, and
  the RRF-fused leg in hybrid). **Fifteen of the sixteen are also the
  vector leg's rank-1 answer**, so they measure whether the keyword leg
  *loses* anything, not whether it *adds* anything. The sixteenth
  (`kw-plant-maintenance-record` → `note-saltmarsh-hide`) is the one that
  measures addition: it is deliberately vector-blind, and its
  authoring rules — including the failed first attempt — are written out
  in the "VECTOR-BLIND KEYWORD TARGET" sections of both fixture files.
  Read those before touching either side of that pair.
- **`paraphrase`** — the query means the same thing as the relevant
  document but shares few or no literal tokens. Built to need semantic
  retrieval; plain-mode recall on this category should stay near zero and
  semantic/hybrid recall should be decisively higher (currently 1.000 vs
  0.000 — see `test_harness_run.py`'s own hard assertion on this).
- **`vocabulary_mismatch`** — a stronger version of paraphrase: the query
  and the document use genuinely disjoint vocabularies for the same concept
  (e.g. "no will" / "intestacy"). **Caveat, found running this harness for
  real, not assumed at design time:** against this corpus and
  `all-MiniLM-L6-v2`, every planted vocabulary-mismatch pair is bridged —
  semantic and hybrid score 1.000 recall at rank 1 on all nine
  `vocabulary_mismatch` queries. Only `plain` shows the gap (0.000). So on
  *this* corpus and model, the category currently measures the
  plain-vs-vector delta, not P2 query-expansion headroom — there is no
  remaining ceiling to detect an *improvement* against, only a regression.
  **P2ab tested exactly that escape hatch and it did not open.** The caveat
  was first written against a 49-document corpus, and the obvious next move
  was "make the corpus harder"; the corpus is now 172 documents with 24 of
  them added specifically as topical anchors and decoys, and this cell is
  still 1.000/1.000. So a *bigger* corpus is not the answer — if P2 work
  wants a category that can show query-expansion gains it needs closer
  distractors than this corpus contains, or a different kind of evidence
  than recall on this fixture set. Do not read vocabulary_mismatch's
  1.000s as "P2 already solved" or "nothing left to build." (The same
  probe-before-you-believe move retired two further P2c premises outright:
  see the unfailable classes in the admission protocol above.)
- **`negation`** — the query asks for the case that *lacks* some property
  ("which outstation does **not** take a standard mains supply"), and the
  answer document describes the exception without ever naming the negated
  aspect ("Skellow Isle draws everything from a solar array"). The cue word
  therefore occurs only in the norm-asserting documents, so the keyword
  paths cannot reach the target and the vector leg is pulled onto the norm.
  **0.000 in all three modes**, and it is the one category whose zero is a
  plain open capability gap rather than a bound on something else. Three
  fixtures, admitted from six probed — see the admission protocol above for
  why the other three are recorded as measured non-failures.
- **`prompt`** — the target is a saved prompt (`source_type = "prompt"`).
  Prompts are **keyword-only by construction**: B2 (TASK-15020) gave them a
  read-only FTS sub-leg and deliberately no vector index, so `semantic`
  reads 0.000 structurally. The category read **0.000 in all three modes**
  until TASK-15400 (2026-08-12); it now reads **0.200 in `hybrid`** — one of
  the five queries — and 0.000 in the other two modes. That one query has
  been rescued by two different mechanisms without the cell moving: the
  stopword trim under 15400's `and_stopword_trim`, and the **prefix
  fallback** under TASK-15700's `and_then_prefix` (2026-08-13), whose full-AND
  primary returns nothing for it. The residual four are a bound on absent
  CONTENT words — not on the cross-sub-leg merge, which 15700 fixed without
  moving them, and not on prompts retrieval: the sub-leg is proven reachable
  on the same runtime. See the
  headroom table and the `docs`/FTS bullet under "Reading the summary table"
  for the measurement and the caution.
- **`negative`** — no relevant document exists for the query. Excluded from
  every averaged precision/recall/MRR/NDCG/F1 (recall over an empty
  relevant set is 0.0 by convention and would drag every average toward
  zero for reasons unrelated to retrieval quality). Reported separately per
  mode instead: `docs_at_k`/`rows_returned` (did the mode return anything
  at all) and `top_score`/`top_vector_score` (how confident the top hit
  was). **These negative-probe numbers are report-only and never gate the
  run** — a junk-similarity regression on a negative query does not fail
  the harness, it is visible in `metadata.report_only.negatives` for a
  human to read. One trap when reading them across a re-stamp: hybrid's
  `top_score` is a *fused RRF* number whose whole scale is `1/(rrf_k+1)`,
  so TASK-4110's k change moved it 0.0115 → 0.1167 without any retrieval
  getting more confident. `max_top_vector_score` stayed at 0.2387 across
  that same re-stamp, and it is the one to read for confidence.
- **`scoped`** — the query runs under a real retrieval scope. It is the one
  category that carries `scope_slugs` (and the only one allowed to): the
  runner translates those fixture slugs into the production
  `EffectiveScope` object, using the runtime ids the real writers assigned,
  and passes it to the seam's own `scope=` parameter. Scoped queries are
  **excluded from the cross-mode overall row** — the same treatment
  negatives get, but for a different reason than either negatives or the
  one this bullet used to give: a scoped query is asked over the hundred
  documents of its scope, while every other query is asked over the whole
  172-document corpus. Those are two different questions, and an average
  over both answers neither. They are still measured, in their own `scoped`
  cell, and each scoped query records **which route actually executed**
  (`runtime_backend` plus the seam's `route_notes` disclosure) — that record
  is what made B1's routing change visible as a change in the report rather
  than as a number that moved.

  *Routing, before and after.* Until TASK-15020/B1 (2026-08-11) the engine's
  allowlist pushdown was semantic-only, so the seam diverted a hybrid
  profile to the semantic path whenever a scope was active and a scoped
  row's `hybrid` and `semantic` columns were one measurement wearing two
  names. B1 pushed the allowlists into the FTS sub-legs and deleted the
  divert: a scoped query now runs its profile's own route, with no
  disclosure. `test_harness_scoped.py` pins that — it pinned the divert
  first, deliberately, so the change could not land silently, and its three
  routing assertions were flipped when it did.

  *What the flip bought, measured.* The seven shipped scoped fixtures are
  keyword-findable inside their scope (`plain` rank 1 on every one, and
  `fts_rank` 1 in the engine's own leg) and outside the vector leg's top-10
  on every one. Diverted, hybrid scored them 0.000; fused, it scores
  **recall 1.000** (MRR 0.163 — the rescued rows arrive late, at ranks 3-9).

  Two mechanisms, not one, and the difference is worth knowing before
  reading these cells as one number. Per-fixture fusion provenance
  (`metadata["hybrid_fusion"]`, k=10, `alpha` 0.7, vector leg over-fetched
  to `2*k`):

  | fixture | fts_rank | vector_rank | rank @`rrf_k`=5 | @`rrf_k`=60 |
  |---|---|---|---|---|
  | `sc-pump-chamber-inspection` | 1 | — | 9 | miss |
  | `sc-storm-overflow-record` | 1 | 12 | 3 | 1 |
  | `sc-intake-screen-survey` | 1 | — | 9 | miss |
  | `sc-meter-box-key` | 1 | — | 9 | miss |
  | `sc-valve-pit-access` | 1 | 20 | 4 | 1 |
  | `sc-sample-point-sign` | 1 | — | 9 | miss |
  | `sc-duty-board-notice` | 1 | — | 9 | miss |
  | | | **scoped recall** | **1.000** | **0.286** |

  Five are FTS-only and reach the top-10 only because `rrf_k` is 5. Two are
  not FTS-only at all — their targets sit at vector rank 12 and 20, inside
  the over-fetched pool, so they carry both legs and lead the list even at
  `rrf_k=60`. The counterfactual for this class at the old constant is
  **0.286, not 0.000**.

  One float detail decides the ordering. `reciprocal_rank_fusion` computes
  `(1.0 - alpha) * fts_rrf`, and `1.0 - 0.7` is `0.30000000000000004`, so an
  FTS-only row scores exactly `0.05` where the semantic leg's rank 9 scores
  `0.7/14` = `0.049999999999999996`: a **strict win by 6.94e-18**, not the
  tie the paper form (`0.3 * 1/6`) would give. The tie-break convention never
  runs. Inclusion, though, has a real margin — the row displaced is the
  semantic leg's rank 10 at `0.7/15` = 0.0467 — so recall 1.000 is not a
  float artefact; only the 9-versus-10 placement is. The scoped cells move
  with the fusion knobs like any other cell.

## Reading the summary table

```
mode           P@k     R@k     MRR    NDCG      F1   docs    n   mean ms   p95 ms  errors  backend
semantic     0.089   0.804   0.804   0.804   0.160    9.7   46      ...
plain        0.304   0.293   0.304   0.296   0.297    0.3   46      ...
hybrid       0.089   0.848   0.809   0.817   0.161   10.0   46      ...
```

Two columns need context before you read the P/R/MRR/NDCG numbers as
"quality":

- **`docs`** is the mean number of documents each query actually returned.
  Precision here divides by `min(k, len(retrieved))`, not by `k` — so
  plain's 0.3 docs/query and P=0.875-on-keyword means "when it returned
  anything it was almost always right," not "it ranked ten results well."
  Compare `docs` before comparing precision across modes.
- **The keyword (FTS) leg is silent for most of this golden set, and the
  `prompt` category is where that shows — TASK-15400, now measured and
  shipped.** `_escape_fts5_query` built the engine leg's MATCH as an
  implicit AND over EVERY query token (TASK-3995 chose that over
  phrase-quoting), with no plural/singular widening — so a natural-language
  query almost never matched. Measured over this golden set at
  TASK-15020/B2 (2026-08-11): the keyword leg returned **zero rows for 40 of
  the 60 queries**, and its **census was 20 of the 53 non-negative queries**
  — "census" being the queries whose TARGET enters the leg's own top-10,
  which is the number that can turn into a hybrid rescue. It fired only
  where the queries are keyword-shaped (`keyword` 13/16, `scoped` 7/7;
  `paraphrase` 0/13, `vocabulary_mismatch` 0/9, `negation` 0/3,
  `prompt` 0/5).

  **TASK-15400 swept four constructions against that number and shipped the
  winner (2026-08-12): `and_stopword_trim`. Census 20 → 21 of 53, zero-row
  40 → 39 of 60, `prompt` census 0/5 → 1/5.** The whole movement is one
  query (`pm-vendor-chaser`, blocked solely by the function word "about"),
  and it is deliberately a small number: the dominant cause is
  AND-strictness over **content** words, which no stopword list touches. The
  full four-row table, the disqualifications and the mechanism are in the
  re-stamp section below; the alternatives measured beside it were
  `build_fts_match_query` (the Library four-seam path's construction —
  rescues 1), `NEAR` proximity (rescues 0, and can only ever narrow), and
  prefix matching (rescues 3, held back by the pre-registered promotion bar
  and carrying the same displacement risk, unmeasured at leg level).

  **TASK-15700 then fixed the merge those disqualifications ran through,
  re-ran the sweep over six rows, and moved the default again (2026-08-13):
  `and_then_prefix` — census 21 → 23 of 53 (20 → 23 against the pre-arc
  control), zero-row 39 → 36 of 60, `prompt` census 1/5 (unmoved, new
  mechanism).** The prefix lead above is what got measured at hybrid level
  and promoted, as a FALLBACK rather than as the primary: the two extra
  census hits are `kw-quillon-mast` and `kw-thimble-relay`, both inflection
  misses ("tension" vs "tensions", "swap" vs "swapped"). `and_then_prefix`
  ships by OWNER RULING over the rule's own tie-break, which selected the
  bare `prefix` row — see the TASK-15700 section above for the full record,
  the 220-statement price, and why zero gated cells moved.

  For media, notes and conversations the semantic leg covers all of this
  completely. Prompts have no semantic leg — B2 gave them an FTS sub-leg and
  deliberately no vector index — so the `prompt` category reads 0.000 in
  `semantic`, and reads 0.200 in `hybrid` on the strength of that single
  rescued query. Do not read a `prompt` 0.000 as a prompts-retrieval defect,
  and do not read the other categories' hybrid numbers as evidence that the
  keyword leg is contributing to them.

  **`plain`'s 0.000 has a DIFFERENT cause, and this paragraph used to give
  the wrong one** (corrected 2026-08-18, TASK-18255). The missing vector
  index cannot explain `plain`, which never consults a vector index at all.
  `plain` fans out over the Library's own four seams, and the harness
  **deliberately does not wire the prompts one** — its fake app sets
  `prompt_scope_service=None`, so `_search_prompts` returns `(False, [])`:
  the seam reporting itself UNAVAILABLE. The cell is therefore **vacuous by
  construction**, not a measured zero, and production does wire the service
  (`app.py:5682`). The warning above was right for the wrong reason in
  `plain`'s case — and TASK-17855 nonetheless filed the defect this sentence
  warns against, because a `0.000` renders identically whether it means "not
  measured" or "measured and found nothing".
- **Plain's MRR and NDCG track recall, not ranking.** The four-seam keyword
  path deliberately drops the FTS rank ("an FTS ranking artifact, not a
  retrieval similarity score" — every plain row carries `score=None`), and
  the seams are merged by a rank-fair rotation (TASK-16071 replaced the old
  fixed-order concatenation; each seam's rank-1 row, then each seam's rank-2
  row, …) rather than by any cross-seam relevance signal. Rank position
  within a seam is the only comparable cross-source signal there is, and it
  is not a score. There is no ranking signal to be right or wrong
  about in plain mode, so its MRR/NDCG columns are recall in a different
  unit. They are still gated (they move when recall moves and catch the
  same regressions), but do not present them as a ranking-quality claim in
  a write-up.
- **Latency (`mean ms`/`p95 ms`) is reported, never gated, and swings hard.**
  Task 6 measured the same run's latency aggregates move 1.7-2.2x on
  process order alone (which `EvalRuntime` runs first in a shared process).
  It lives in `metadata.report_only.latency`, is rounded to 0.1s
  specifically to keep re-stamp diffs from churning on noise, and is never
  part of the fingerprint or the gate. Do not add a latency threshold to
  this harness without first re-measuring that variance — it will fail on
  noise, not on a real slowdown.
- **40 of the 105 gated cells sit at exactly 0.000**, and you should know
  which before quoting the gate's coverage. A metric already at its floor
  cannot register a regression — there is nowhere lower for it to go — so
  those cells are structurally inert to the gate today. They are: `plain`
  20 (paraphrase, vocabulary_mismatch, negation, prompt), `semantic` 15
  (negation, prompt, scoped), `hybrid` **5** (negation) — all five gated
  metrics each. **This was 45 after P2ab and is 40 after TASK-15400**: the
  construction flip took hybrid's five `prompt` cells off the floor, which
  is what taking headroom looks like in this table. It is still up from 10
  of 60 before P2ab, and that increase is the deliberate cost of fail-first
  authoring: a cell at 0.000 is exactly what a category with headroom looks
  like before the headroom is taken. The live (non-floor) count has risen
  50 → 60 → **65**, so the gate watches more than it did; it just also
  carries more declared-empty cells that P2c is meant to fill.
  If plain-mode keyword expansion (P2) ever gives those cells a nonzero
  value, the baseline must be re-stamped before the *next* change can be
  gated against them; until then, treat "plain paraphrase precision is
  still 0.000" as expected, not as evidence the gate is watching that cell.

## The doc-level canonicalization contract

Retrieval is measured **per document**, but the seam returns **rows**: a row
is one chunk in semantic/hybrid mode and one whole item in plain (keyword)
mode, and a single document can occupy several of the top-k slots.
`harness/canonicalize.py` collapses seam rows to the fixture slugs the
golden set is written in — the golden set never names a database id,
because ids are assigned per-run by the real writer APIs — keeping each
document's *first-hit* rank and deduping the rest.

The contract, in short:

- A row is resolved by `(canonical_source_type(row.provenance.source_type),
  str(row.source_id))` against a `slug_to_source` lookup the ingestion
  runtime built while writing the fixtures.
- Source types arrive in more than one vocabulary across the five row
  builders (`note`/`notes`, `conversation`/`conversations`/`chat`,
  `media`/`media_chunk`) and are folded to the app's singular ITEM_TYPE_*
  form before lookup.
- The hybrid FTS leg used to be a special case: its `source_id` was not a
  bare source id but `SearchResult.id` (e.g. `media_15`, a *document* id,
  not a *source* id), because `RAGService._keyword_search` never populated
  the `source_id` key `_semantic_row` reads. TASK-3996 made all three
  sub-legs stamp a bare `source_id` — that vocabulary equality is exactly
  what lets a keyword row fuse with its vector twin — so the prefixed form
  should no longer arrive here. The prefix-stripping retry in
  `canonicalize.py` is kept anyway: it costs nothing, and it is the
  difference between a future row builder being *measured wrong* and being
  measured. The raw id is kept when the retry also misses, so this is a
  lookup fallback, never a silent rewrite of some other row's identity.
- **A row that resolves to no fixture document is kept, not dropped**, as
  `"unknown:<type>:<id>"`. Dropping it would make precision answer "of the
  documents I recognized, how many were right" — a number that improves
  when retrieval returns more garbage. It occupies a top-k slot in the real
  product, so it occupies one here too.

If you add a new row builder in the seam, or a new source type, extend
`SOURCE_TYPE_ALIASES` in `harness/canonicalize.py` and add a
`test_canonicalize.py` case before trusting any metric this harness produces
for that path.

## Known, deliberately-recorded defects

The first real harness run surfaced four real defects in the Library/RAG
retrieval seams — not harness bugs. Closing three of them (the fusion
cluster, re-stamped in one commit — see the progression table above)
surfaced a fifth, TASK-4110, which is now fixed and re-stamped too. Read
the task descriptions for exact mechanisms and source locations before
attempting anything here:

- **TASK-3995 — FIXED.** The engine's keyword leg wrapped every query in
  FTS5 phrase quotes, so multi-token queries required a contiguous token
  run. It now quotes each token individually and joins them with FTS5's
  implicit AND, which keeps the injection safety that made whole-query
  quoting attractive while dropping the phrase semantics.
- **TASK-3994 — FIXED.** Fusion matched the two legs on
  `SearchResult.id` across mismatched id spaces (`media_15` vs
  `media_15_chunk_0`), so a document found by both legs could never fuse.
  It now matches on `(source_type, source_id)` document identity, and a
  merged row displays the vector leg's matched chunk. Hybrid is no longer
  byte-identical to semantic (22/44 golden queries differed immediately
  after the fix). Its second AC — hybrid *rescuing* a document the vector
  leg misses — was split out and closed later, by TASK-4110's weighting
  change; `kw-plant-maintenance-record`'s hybrid cell going miss → hit is
  what closed it.
- **TASK-3996 — FIXED.** The keyword leg only joined `Media`/`media_fts`.
  It is now three read-only sub-legs (media, notes, conversations) merged
  by rank-position interleaving, so notes and conversations are reachable
  through hybrid search. Verified live as well as here: with Media
  deselected, a three-token non-contiguous query returned a note row and a
  conversation row banded `keyword match`.
- **TASK-3997 — open.** Investigation: the four-seam (plain) keyword path
  AND-joins every query term group, so one term with no match anywhere in
  the corpus zeroes the whole query's result set. An
  investigation/product-judgment task, not a defect with an obvious fix.
  **The two keyword paths now diverge deliberately, and this is the record
  of it (TASK-15400 AC#8).** The ENGINE leg builds its MATCH through
  `RAGService._fts5_match_expressions` and ships `and_then_prefix`
  (TASK-15700, 2026-08-13; `and_stopword_trim` before that); the
  Library FOUR-SEAM path still builds its own through
  `build_fts_match_query` and still AND-joins every group. They were not
  unified, for a measured reason rather than a scheduling one: the sweep
  that chose the engine's construction is a *hybrid-fusion* measurement —
  every constraint that decided it (the vector-blind rescue, the scoped
  collapse) is about rows competing inside a fused top-k, and the four-seam
  path has no fusion, no relevance ranking (TASK-16071's merge orders it by
  within-seam rank position, which is not a score) and no leg to be
  displaced in. Its
  construction therefore has to be decided on its own evidence, which is
  what TASK-3997 is for; `build_fts_match_query` was measured alongside the
  four swept rows (it rescues 1 of the 40 zero-row queries, the same order
  as the winner) so that decision starts from a number.
- **TASK-4110 — FIXED, found by this harness closing the cluster.** An
  FTS-only row could not enter hybrid's fused top-k. The fused score is
  `(1-alpha)/(rrf_k+fts_rank) + alpha/(rrf_k+vector_rank)`; at the old
  defaults (alpha 0.7, rrf_k 60) a keyword rank-1 row scored 0.00492 and
  lost to every vector row better than rank ~82, while the vector leg is
  only ever asked for `top_k * hybrid_pool_multiplier` (~2k) results. So
  whenever the vector leg returned k or more **distinct documents**, a
  document only the keyword leg found was structurally unreachable.
  Distinct *documents*, not rows, is the load-bearing word: fusion dedups
  by document identity, so a vector leg returning `2k` chunk rows that
  collapse to fewer than k documents leaves room — which is why
  keyword-only rows appeared anyway when the vector index was thin, and why
  such a sighting is **not** evidence the starvation is absent. The fix was
  a measured retune of one constant, `rrf_k` 60 → **5**
  (`RAG_Search/simplified/config.py`'s `DEFAULT_HYBRID_RRF_K`), chosen from
  a six-strategy sweep (`test_fusion_sweep.py`, still runnable) rather than
  asserted: at k=5 an FTS-only rank-1 row strictly outranks vector-only
  rows from rank 10, which is inside the ~20-row window fusion actually
  sees. `kw-plant-maintenance-record` is the before/after: plain rank 1,
  semantic absent, engine FTS leg rank 1, hybrid **absent → rank 8**.
  Alpha (0.7) and the pool multiplier (2) were measured alongside and
  deliberately left alone.
- **TASK-15400 — FIXED, and the fix is smaller than the defect.** The engine
  keyword leg AND-ed every query token, so it returned nothing for 40 of the
  60 golden queries. Four constructions were swept under a pre-registered
  rule and `and_stopword_trim` shipped: census 20 → 21 of 53, zero-row
  40 → **39** of 60, hybrid `prompt` recall 0.000 → 0.200, no cell down in
  any mode, zero extra FTS queries. Read that as what it is — the arc moved
  one query. **The candidates that move the census properly (28-29, prompt
  5/5) are exactly the ones the constraints reject, and they are rejected by
  the MERGE rather than by the match form** — see the re-stamp section
  above, and TASK-15700, which owns the round-robin `interleave_rankings`
  displacement that disqualified them.
- **TASK-15700 — FIXED, and it changed which mechanism the leg is bounded
  by.** `RAGService._keyword_search` merged its four source sub-legs with a
  round-robin over sub-leg POSITION, so one sub-leg's row count decided
  another sub-leg's leg rank and fusion consumed that rank. The merge is now
  form-tiered (primary-form rows wholly precede fallback-form rows), which
  restored `and_then_or`'s vector-blind rescue (`NO` → yes at slot 10) and
  its scoped recall (0.429 → **1.000**) — the predicted zero-headroom rescue,
  to the slot. The re-run sweep then moved the default to **`and_then_prefix`
  by owner ruling** (the rule's own tie-break selected `prefix`): census
  20 → 23 of 53, zero-row 40 → **36** of 60, **0 of 105 gated cells moved**,
  at 220 extra FTS statements over the 60 golden queries. `and_then_or` is
  still disqualified — now on constraint (b), by FUSION (tier-2 fallback rows
  carrying vector ranks promote to merged rows above rank-1 answers), and
  Part A is provably inert on those queries because their keyword legs are
  100% tier 2. The full record is the TASK-15700 section above.

Do not "fix" any of these by editing the harness or the fixtures — the
harness's job is to keep measuring the real seam accurately; the numbers it
reports for hybrid mode *are* today's truth about that seam.

**Two fusion paths, one k, two identity rules.** The engine's
`_fuse_hybrid_results` keys on `(source_type, source_id)`, while
`pipeline_builder`'s parallel hybrid path keys on `(source, id)` — the same
`reciprocal_rank_fusion` helper, two different notions of "the same
document". They no longer disagree about the *weighting*: both resolve k
through `fusion.resolve_rrf_k`, which falls back to the active profile, so
TASK-4110's measured `rrf_k = 5` moved both live paths together even though
only the engine path was measured (deliberate, and pinned by
`Tests/RAG/test_fusion.py::TestPipelineRrfMerge`). The identity rules still
differ, and only the engine path is measured here: reconciling them is
TASK-3501, and until then a hybrid *quality* number from this harness
describes the engine path and does not transfer to the pipeline path.

## Adding a fixture document or golden query

1. Add a `[[doc]]` entry to `fixtures/corpus.toml` (`slug`, `source_type`
   ∈ `note`/`media`/`conversation`, `title`, `content`) or a `[[query]]`
   entry to `fixtures/golden.toml` (`id`, `query`, `category`,
   `relevant_slugs` — a list of corpus slugs, `[]` only for `category =
   "negative"`; plus `scope_slugs` for `category = "scoped"`, which only
   that category may carry and every scoped query must — a non-empty list
   of `media`/`note` slugs that includes all of its own `relevant_slugs`.
   Conversations cannot be scoped: they are outside the app's scope
   vocabulary).
2. Run `pytest Tests/RAG_Eval/test_goldenset_integrity.py -q` — the
   validator fails fast and lists **every** structural defect at once
   (duplicate slugs/ids, an unknown category or source_type, a
   `relevant_slugs` entry with no matching document, a category or source
   type with zero members). Fix everything it reports before moving on.
   Two of those tests pin the exact corpus/golden-set *sizes*
   (`test_corpus_composition_matches_the_planned_design`,
   `test_golden_set_category_quotas`), so an addition is never accidental:
   update them in the same commit, with the reason for the new document in
   the assertion's comment.
3. If you touched `corpus.toml` or `golden.toml`, the fixture SHA-256 in
   `current_fingerprint()` changes, which means the *next* gated run will
   report `ENVIRONMENT_CHANGED` against the committed baselines rather than
   scoring anything — re-stamp deliberately (see above), with the new
   fixture content named as the reason in the PR, not silently absorbed
   into an unrelated change's re-stamp.

## This corpus as an ANSWER-level instrument (TASK-16174, 2026-08-15)

Everything above measures **retrieval**: whether the right document
arrives, scored rank-wise at the document level. TASK-16174 needed a
different question — *what happens after it arrives?* — and reused this
corpus for it, without touching a fixture, a golden query, or a baseline.
The gated suite read **`[rag-eval baselines] PASSED: No regression. 105
metric(s) within 0.05 of baseline.`** with all 105 cells at `(+0.000)`
before and after the arc; that is the proof the answer-level work changed
no retrieval behaviour.

**What the arc shipped.**

- **Phase K — the knobs retired.** `SearchConfig.include_parent_docs`,
  `parent_size_threshold` and `parent_inclusion_strategy` were deleted,
  along with nine writes across three shipped RAG profiles. A grep had
  proved zero reads: they were a user-switchable surface wired to nothing.
  `RAGConfig.from_dict` now filters unknown `[search]` keys with a logged
  notice, so a saved config still carrying them loads instead of raising
  `TypeError`. (`Docs/Development/CHUNKING-*.md` carried the false claim
  that this feature was implemented; those documents now carry a
  correction.)
- **Phase T — the tool.** `expand_document`
  (`tldw_chatbook/Tools/document_expansion_tool.py`), a gateable built-in
  behind `[tools] expand_document_enabled` (OFF by default) and
  `risk_tags=("reads",)`, which floors it to *ask*. It takes exactly what
  a row carries — `source_type` + `source_id` required, plus optional
  `chunk_start` (the window anchor) and `offset` (continuation) — and
  returns a budgeted window of the note / media / conversation / prompt
  behind the hit. `chunk_id` is NOT a parameter: it is an index
  (`f"{doc_id}_chunk_{i}"`), nothing in the tool reads it, and the final
  review found it shipped as agent-facing schema wired to nothing — the
  fix wave retired it (a pasted row still works: it rides the
  `**_provenance` swallow).
- **Phase P — the policy, wired.** `Library/library_expand_policy.py`
  computes a per-row `{expandable, reason}` verdict, and
  `Agents/library_rag_tool_provider._project_row` attaches it *plus* the
  `source_type`/`source_id` the tool requires, under exactly the hint's
  own precondition. The agent is told which rows are labels rather than
  left to infer it, and is given an identity it can act on rather than
  one it must guess. A row also carries `chunk_id` when it has one and
  `chunk_start` **when its provenance carries a usable anchor** (fix wave,
  after the final review): a head anchor (`0`) or an unparseable one is
  dropped, because `expand_document` centres its window only for
  `anchor > 0` and a key that changes nothing is bytes spent for no
  behaviour. Cost re-measured by T3b's strip-and-reserialize method on a
  ten-row payload with five anchored rows: **+19.0 B per anchored row,
  +95 B total (9.5 B/row), 11.7 % of the 32 KiB ceiling**; an unanchored
  payload pays nothing.

**Phase E — the oracle run.** Eight fixed questions over this corpus,
each naming one media or conversation document and carrying a fact-oracle
regex grep-verified to appear only in that document's **body** — never in
a title (the agent sees titles), never in another document, never in the
question. The real agent loop (`AgentService` → `chat_api_call` →
`api.anthropic.com`, `claude-haiku-4-5`, temperature 0) answered each
question twice: once in the shipped default posture (gate off) and once
with the gate on. Scoring is mechanical oracle inclusion in the final
answer — **no LLM grader**. The route is `plain`
(`default_search_mode = "plain"`), which is what emits the label-only
`Matched media · document` / `Matched conversation · N messages` rows the
tool exists for; under `hybrid` there is nothing label-only to expand.

| question | target (label-only row) | oracle | tool-OFF | tool-ON | ON tool calls |
|---|---|---|---|---|---|
| `q1-quillon-access` | `media-quillon-antenna` (media) | `/rescue certification/` | miss | HIT | search_library_rag, search_library_rag, expand_document |
| `q2-pellucid-lag` | `media-pellucid-gauge` (media) | `/lowest decade/` | miss | HIT | search_library_rag, search_library_rag, expand_document |
| `q3-ashgrove-seal` | `conv-ashgrove-pump` (conversation) | `/shimming/` | miss | HIT | search_library_rag, expand_document |
| `q4-obsidian-bearing` | `media-obsidian-lathe` (media) | `/brinelling/` | miss | HIT | search_library_rag, expand_document |
| `q5-dunnock-cooling` | `conv-dunnock-row-cooling` (conversation) | `/blown sand/` | miss | HIT | search_library_rag, search_library_rag, expand_document |
| `q6-gatehouse-ups` | `conv-gatehouse-power` (conversation) | `/(?:ninety\|90)\s*minutes/` | miss | miss | search_library_rag, search_library_rag |
| `q7-filling-head-mtbf` | `media-filling-head-reliability` (media) | `/(?:nine hundred\|900)\s+(?:running\s+)?hours/` | miss | HIT | search_library_rag, search_library_rag, search_library_rag, expand_document |
| `q8-larkspur-lubrication` | `media-larkspur-turbine` (media) | `/starved a bearing/` | miss | HIT | search_library_rag, expand_document |
| **TOTAL** | | | **0/8** | **7/8** | |

Total spend $0.177 (plus $0.022 for a one-question smoke run). The single
tool-ON miss is a **retrieval** failure present in both arms — both of
`q6`'s searches returned `status: "empty"`, so there was never a row to
expand (the four-seam keyword path ANDs every query token, and the model's
own phrasings over-constrained; the document is reachable at rank 1 for
`gate house UPS`). Conditioned on the target row actually being returned,
the tool-ON arm answered **7 of 7**. Four tool-OFF questions retrieved the
label-only row and still scored zero, saying so unprompted: *"I can only
see that it's a media document and cannot access its full contents through
the search results."*

Method, isolation proof, per-question attribution, disclosed limits and
the full artifact:
`Docs/superpowers/qa/2026-08-15-rag-agentic-expansion/` (`questions.toml`,
`oracle_run.py`, `report.md`, `run-artifacts.json`). The run is a
**script, never a test** — the suite is network-blocked by default, and
`oracle_run.py --dry-run` reproduces everything except the model calls.

**If you extend this instrument**, keep two properties or it stops
measuring anything: an oracle must never appear in a title (the agent sees
titles), and the route must stay `plain` unless you are deliberately
measuring a different regime.

## The semantic/hybrid routes, measured (TASK-16588, 2026-08-16)

The Phase E oracle run above is a `plain`-route instrument by construction —
that route emits no chunked rows and its `source_id` is always a real
database id, so it can neither exercise `chunk_start` nor ever need an
identity fallback. TASK-16588 closed that gap with a **mechanical** probe (no
LLM, no network, no spend, no TUI boot) over `semantic` and `hybrid`, against
TWO index kinds — because the identity gap is a property of the INDEX's
metadata vocabulary, not of the route:

* **canonical** — 20 items (12 notes, 4 media, 4 conversations) through the
  app's own `note_document`/`media_document`/`conversation_document` builders
  + `index_entries`. Chunk metadata carries `source_id`/`source_type`, so
  `_semantic_row` resolves a real id.
* **non-canonical** — the same 12 notes through a hand-built `IndexEntry`
  with `{"type","note_id","title"}` metadata (the shape TASK-15810's
  committed QA seeder writes). No `source_id`, so `_semantic_row` falls
  through to the vector store's POINT id.

Rows come from the production surface (`LibraryRagToolProvider.invoke`,
`mode="rag"`, the sealed 32 KiB payload). Every identity-bearing row is then
expanded by a DIRECT `ExpandDocumentTool().execute(...)` call in three arms
over the SAME row: `pre` (the `note_id`/`doc_id` keys stripped — the payload
as it was before this task, without a checkout dance), `post` (as shipped),
and `head` (`post` minus `chunk_start`, the control).

| index × route | rows | hinted | expandable | not_found PRE (hinted / expandable) | not_found POST | `chunk_start` carried | marker windows (post / head) | variant rows w/o hint |
|---|---|---|---|---|---|---|---|---|
| canonical × semantic | 100 | 100 | 64 | 0 / 0 | 0 / 0 | 69 | 7/7 · 0/7 | 0 |
| canonical × hybrid | 100 | 100 | 61 | 0 / 0 | 0 / 0 | 56 | 7/7 · 0/7 | 0 |
| non-canonical × semantic | 70 | 70 | 49 | **70 / 49** | 0 / 0 | 45 | 4/4 · 0/4 | 0 |
| non-canonical × hybrid | 70 | 70 | 45 | **66 / 45** | 0 / 0 | 43 | 4/4 · 0/4 | 0 |

All three pre-registered expectations held. **The window question is answered
for the first time on any route:** 22 of 22 rows whose matched chunk carried
the planted marker returned a `chunk_start`-anchored window CONTAINING it
(marker planted 9,624–9,736 chars into a ~12,300-char document, past
`expand_document`'s 8,000-char budget), and 0 of 22 head windows did — the
control that makes the check failable. **The identity question is answered
too:** on a non-canonical index every row the hint declared expandable came
back `not_found` before the fallbacks shipped and `ok` after; on a canonical
index the reading was 0 both before and after, so the fix is
defensive-plus-legacy and the report says so rather than overclaiming.

A broader per-row check ran in every arm and is the stronger AC#3 reading:
does the returned window contain the first 160 chars of that row's OWN
snippet? Over all 340 rows — **340 / 340 post-fix**, 186/340 on the `head`
(anchor-stripped) arm, and on the `pre` (fallback-stripped) arm 200/200
canonical but only **4 / 140** non-canonical. The post-fix 340/340 proves the
expansion resolved the RIGHT document on every rescued row, not merely *a*
document; the head arm's 154 failures make the anchor control failable across
all 340 rows rather than the 22 marker ones.

Byte cost, re-measured by strip-and-reserialize on the 34 real route
payloads: **45.94 B per carrying row canonical (a redundant `doc_id`), 102.0
B non-canonical (`note_id` + `doc_id`)** — 3–7× the +15.0 B/row the unit
fixture reports, because real ids are UUIDs and the fixture's were `n1`.
Largest payload 17,350 B of 32,768 (53 %); `returned == 10` on every payload,
so the sealing loop dropped nothing.

**If you extend this instrument**, keep the property that makes it work: the
marker must sit past the tool's default budget, or the anchored-window check
becomes unfailable, and the head-window arm is what proves it has not. And
know the limit this run does NOT escape: all 22 anchored windows are the
document **TAIL** (`[total − 8000, total]`), because a ~12.3k-char document
and an 8,000-char budget leave a `chunk_start` of ~9,200 only two reachable
outcomes, head or tail. 22/22 therefore proves "off the head", not "centred
on the match". To show a true mid-document slice, make the document 3–5× the
budget AND plant markers past the budget but NOT within one budget of the
tail (`budget/2 < chunk_start < total − budget/2`). Two more things this
probe deliberately does NOT measure: `label_only` rows (0 of 340
— they are a `plain`-route product of the Library's four-seam keyword path,
which is Phase E's regime), and retrieval quality (every marker query put its
target at rank 1 by design; the gated suite is the instrument for that, and
it read 105/105 at (+0.000) throughout).

The table's last column is the answer to TASK-16174's finding 6:
**canonicalization-VARIANT rows (`notes`, `media_chunk`, `conversations`,
`chat`, `prompts` — spellings `_SEMANTIC_SOURCE_TYPE_MAP` treats as live but
`library_expand_policy.EXPANDABLE_SOURCE_TYPES` does not) read 0 of 340**,
against a committed positive control showing the same `expand_hint` helper
fires on every variant and on no singular; on that measured zero TASK-16688
RECORDED the exclusion (a module docstring note plus a both-directions pin in
`Tests/Library/test_library_expand_policy.py`) rather than broadening the
allowlist for a producer that does not exist.

Method, isolation proof, per-row detail and the full artifact:
`Docs/superpowers/qa/2026-08-16-rag-semantic-identity/` (`route_probe.py`,
`report.md`, `probe-artifacts.json`). The probe is a **script, never a test**
— it builds two real embedding indexes in two scratch profiles.
