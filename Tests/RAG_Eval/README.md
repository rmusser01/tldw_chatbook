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

> **Reading this during the P2ab arc.** The fixture set has been renewed
> (172 documents, 60 queries) with fail-first classes — `scoped`,
> `negation`, `prompt` — admitted only where today's pipeline was measured
> to fail them. The per-category analysis further down still describes the
> pre-renewal state, in which hybrid recall sat at 1.000 everywhere; the
> per-category headroom table that replaces it, and the class-level
> admission outcomes (`compositional` and `acronym` proved unfailable on
> this corpus and model), land with the arc's one deliberate re-stamp. Until
> then every gated run reads `environment_changed`, which is expected and
> not a regression. See golden.toml's fail-first block for the measurements.

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
  conftest.py                      un-sandboxes the model cache dir for env-gated tests only
  harness/
    environment.py                 the RAG_EVAL gate + HF-offline latch
    goldenset.py                   fixture loader/validator (corpus.toml, golden.toml)
    ingest.py                      corpus -> real source DBs -> isolated indexed RAGService
    canonicalize.py                seam rows -> fixture-slug document ids
    runner.py                      the three-mode run + per-category scoring
    baseline_io.py                 fingerprinted baselines + the fail-on-regression gate
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
currently 138 passed, 4 skipped (the two env-gated test files, two tests
each). It never touches the embedding model or a real corpus.

**The real harness run (opt-in, slow, needs a real model):**

```bash
RAG_EVAL=1 pytest Tests/RAG_Eval/ -q -p no:randomly
```

This runs everything above *plus* `test_harness_smoke.py` and
`test_harness_run.py`, which stand up a genuine indexed RAG install (the 49
fixture docs, written through the real writer APIs, embedded and indexed
through the real batch path) and run all 45 golden queries three times —
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
every one of the 60 gated metrics old -> new (or `absent -> value` on first
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

**The rescue query is now at its ceiling too.** With
`kw-plant-maintenance-record` answered, every scored query in this corpus
is a hit in hybrid mode (overall recall 1.000). A future fusion retune
therefore has *nothing left to improve* here and can only be measured for
regression — the same trap documented for `vocabulary_mismatch` above. If
another weighting change needs evidence that it *adds* something, it needs
a **new** vector-blind fixture authored the way the existing one was (see
the "VECTOR-BLIND KEYWORD TARGET" sections in both fixture files); do not
read this corpus going quiet as "fusion is finished".

**The bound on "nothing regressed".** It is a 49-document corpus in which
every scored query except one was already answered at rank 1, so there was
very little left to damage; and this harness measures at `k = 10`, while
the Library Search/RAG surface defaults to `LIBRARY_RAG_DEFAULT_TOP_K = 5`
(half the fused candidate window, since `_hybrid_search` fetches
`top_k * hybrid_pool_multiplier` per leg). A hybrid number here is a
statement about this corpus at k=10, not a promise about a large library at
k=5. TASK-4110's live check answered the k=5 half separately, against a
real 64-document library through the running TUI.

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

Every golden query carries one of five categories:

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
  real, not assumed at design time:** against this 49-document corpus and
  `all-MiniLM-L6-v2`, every planted vocabulary-mismatch pair is bridged —
  semantic and hybrid score 1.000 recall at rank 1 on all nine
  `vocabulary_mismatch` queries. Only `plain` shows the gap (0.000). So on
  *this* corpus and model, the category currently measures the
  plain-vs-vector delta, not P2 query-expansion headroom — there is no
  remaining ceiling to detect an *improvement* against, only a regression.
  If P2 work wants a category that can show query-expansion gains, it needs
  either a harder corpus (closer topical distractors, longer documents) or
  a different kind of evidence than recall on this fixture set; do not read
  vocabulary_mismatch's current 1.000s as "P2 already solved" or "nothing
  left to build."
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
semantic     0.114   0.974   0.974   0.974   0.203    9.1   38      ...
plain        0.368   0.355   0.368   0.358   0.360    0.4   38      ...
hybrid       0.103   0.974   0.974   0.974   0.185   10.0   38      ...
```

Two columns need context before you read the P/R/MRR/NDCG numbers as
"quality":

- **`docs`** is the mean number of documents each query actually returned.
  Precision here divides by `min(k, len(retrieved))`, not by `k` — so
  plain's 0.4 docs/query and P=0.875-on-keyword means "when it returned
  anything it was almost always right," not "it ranked ten results well."
  Compare `docs` before comparing precision across modes.
- **The keyword (FTS) leg is silent for most of this golden set, and the
  `prompt` category is where that shows.** `_escape_fts5_query` builds the
  engine leg's MATCH as an implicit AND over EVERY query token, function
  words included (TASK-3995 chose that over phrase-quoting), with no
  plural/singular widening — so a natural-language query almost never
  matches. Measured over this golden set at TASK-15020/B2 (2026-08-11): the
  keyword leg returns **zero rows for 40 of the 60 queries**, firing only
  where the queries are keyword-shaped (`keyword` 13/16 targets found by
  the FTS leg alone, `scoped` 7/7; `paraphrase` 0/13,
  `vocabulary_mismatch` 0/9, `negation` 0/3, `prompt` 0/5). For media,
  notes and conversations the semantic leg covers that completely. Prompts
  have no semantic leg — B2 gave them an FTS sub-leg and deliberately no
  vector index — so the `prompt` category reads 0.000 in all three modes
  even though the sub-leg works: the same runtime answers the
  keyword-shaped "shift log summary supervisor" with the right prompt at
  hybrid rank 9, as an FTS-only row. Do not read a `prompt` 0.000 as a
  prompts-retrieval defect, and do not read the other categories' hybrid
  numbers as evidence that the keyword leg is contributing to them.
- **Plain's MRR and NDCG track recall, not ranking.** The four-seam keyword
  path deliberately drops the FTS rank ("an FTS ranking artifact, not a
  retrieval similarity score" — every plain row carries `score=None`), and
  rows arrive concatenated in a fixed seam order (notes, media,
  conversations, prompts). There is no ranking signal to be right or wrong
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
- **Ten of the `plain` cells sit at exactly 0.000** (paraphrase and
  vocabulary_mismatch, all five gated metrics each — precision, recall,
  MRR, NDCG and F1 — across the categories where plain-mode has no
  literal-token overlap to find). A metric already
  at its floor cannot register a regression — there is nowhere lower for it
  to go — so those specific cells are structurally inert to the gate today.
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
