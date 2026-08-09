# RAG_Eval — the P1 retrieval eval harness

This directory is the measuring instrument for `tldw_chatbook`'s Library
retrieval seam (`LibraryLocalRagSearchService`). P0 (task-3170) made
profile-driven retrieval reachable from Console and Library search but
deliberately made no quality claims. P1 (TASK-3894) builds the thing that
can make those claims: ported precision/recall/MRR/NDCG/F1@k metrics, a
regression/gating layer with fingerprinted committed baselines, a
deterministic fixture corpus + golden query set spanning notes, media and
conversations, and an env-gated pytest harness that runs every golden query
through the real seam across all three profile modes.

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
  conftest.py                      un-sandboxes the model cache dir for env-gated tests only
  harness/
    environment.py                 the RAG_EVAL gate + HF-offline latch
    goldenset.py                   fixture loader/validator (corpus.toml, golden.toml)
    ingest.py                      corpus -> real source DBs -> isolated indexed RAGService
    canonicalize.py                seam rows -> fixture-slug document ids
    runner.py                      the three-mode run + per-category scoring
    baseline_io.py                 fingerprinted baselines + the fail-on-regression gate
  fixtures/
    corpus.toml                    48 fixture documents (note/media/conversation)
    golden.toml                    44 golden queries (37 scored, 7 negative)
  baselines/
    semantic.json, plain.json, hybrid.json   committed, fingerprinted, per-mode baselines
```

The metrics, regression/gating, fixture-integrity and canonicalization
modules are ported from `tldw_chatbook/RAG_Search/eval/` and are always-on:
they need no model, no extras, and run in every ordinary `pytest` invocation.
Only the two files that stand up a real indexed corpus and run real queries
through the real seam (`test_harness_smoke.py`, `test_harness_run.py`) are
gated behind `RAG_EVAL=1`.

## Running it

**Always-on (no gate, no model, part of the normal suite):**

```bash
pytest Tests/RAG_Eval/ -q
```

This runs the metrics/gating/fixture-integrity/canonicalization tests —
currently 128 passed, 4 skipped (the two env-gated test files, two tests
each). It never touches the embedding model or a real corpus.

**The real harness run (opt-in, slow, needs a real model):**

```bash
RAG_EVAL=1 pytest Tests/RAG_Eval/ -q -p no:randomly
```

This runs everything above *plus* `test_harness_smoke.py` and
`test_harness_run.py`, which stand up a genuine indexed RAG install (the 48
fixture docs, written through the real writer APIs, embedded and indexed
through the real batch path) and run all 44 golden queries three times —
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

**This will happen for real, soon.** TASK-3994 (hybrid RRF fusion never
merges its legs) records today's truth: hybrid search is byte-identical to
semantic search on this corpus (44/44 identical id-lists), because the two
legs fuse on mismatched id spaces and the FTS leg's contribution never
survives into the fused top-k. `hybrid.json`'s baseline encodes that defect.
**Fixing TASK-3994 will change hybrid's numbers and will trip this gate —
that is correct, expected behavior, not a new bug.** The re-stamp belongs in
the same PR as that fix, with both the "hybrid ≡ semantic" before-numbers
and the post-fix after-numbers in the PR description.

## Fingerprint semantics: "environment changed" is not a regression

Every baseline carries an environment fingerprint: the embedding model
string, the installed `sentence-transformers` version, a SHA-256 over both
fixture files (length-delimited, so moving a byte between them cannot
produce the same digest), and `sys.platform`. A `pipeline_config` block
(`k`, profile name, `source_types`) is compared alongside it, prefixed
`pipeline_config.` in any diff, so a `k` change reads distinctly from an
embedding-environment change.

When the current run's fingerprint (or pipeline config) does not match the
committed baseline's, the gate reports `ENVIRONMENT_CHANGED` and **does not
score the run at all** — it is not a pass because retrieval held up, it is a
pass because the numbers were never comparable in the first place. This
matters concretely: these baselines were stamped on `darwin` with
`sentence-transformers 5.4.1`. On a different platform or a different
`sentence-transformers` version, the gate will go green having checked
*nothing* — it prints a `NOTE:` naming the differing keys and asking you to
re-stamp on that machine. Read that note; do not read the green exit code
alone as "retrieval is fine here."

A missing baseline is treated as a **failure**, not a pass — the same
reasoning as pytest's own "no tests ran": a green gate that checked nothing
is more dangerous than a red one.

## Category meanings

Every golden query carries one of four categories:

- **`keyword`** — the query shares literal tokens with its relevant
  document(s). Exercises the FTS/keyword leg (plain's four-seam path, and
  the RRF-fused leg in hybrid).
- **`paraphrase`** — the query means the same thing as the relevant
  document but shares few or no literal tokens. Built to need semantic
  retrieval; plain-mode recall on this category should stay near zero and
  semantic/hybrid recall should be decisively higher (currently 1.000 vs
  0.000 — see `test_harness_run.py`'s own hard assertion on this).
- **`vocabulary_mismatch`** — a stronger version of paraphrase: the query
  and the document use genuinely disjoint vocabularies for the same concept
  (e.g. "no will" / "intestacy"). **Caveat, found running this harness for
  real, not assumed at design time:** against this 48-document corpus and
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
  human to read.

## Reading the summary table

```
mode           P@k     R@k     MRR    NDCG      F1   docs    n   mean ms   p95 ms  errors  backend
semantic     0.117   1.000   1.000   1.000   0.208    9.1   37      ...
plain        0.351   0.338   0.351   0.341   0.342    0.4   37      ...
hybrid       0.117   1.000   1.000   1.000   0.208    9.1   37      ...
```

Two columns need context before you read the P/R/MRR/NDCG numbers as
"quality":

- **`docs`** is the mean number of documents each query actually returned.
  Precision here divides by `min(k, len(retrieved))`, not by `k` — so
  plain's 0.4 docs/query and P=0.867-on-keyword means "when it returned
  anything it was almost always right," not "it ranked ten results well."
  Compare `docs` before comparing precision across modes.
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
- The hybrid FTS leg is a special case: its `source_id` is not a bare
  source id but `SearchResult.id` (e.g. `media_15`, a *document* id, not a
  *source* id), because `RAGService._keyword_search` never populates the
  `source_id` key `_semantic_row` normally reads. A lookup miss is retried
  once with the leading `f"{source_type}_"` stripped, and the raw id is
  kept when that retry also misses — so this is a lookup fallback, never a
  silent rewrite of some other row's identity.
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
retrieval seams — not harness bugs, and not fixed as part of P1, because
fixing them is P2 scope and each fix needs its own before/after baseline
re-stamp (see "The P2 discipline" above). They are filed as backlog tasks;
read the task descriptions for exact mechanisms and source locations before
attempting a fix:

- **TASK-3994** — hybrid RRF fusion never merges its legs (id-space
  mismatch between the FTS leg's document ids and the vector leg's chunk
  ids); hybrid is presently byte-identical to semantic on this corpus.
- **TASK-3995** — the engine's keyword leg wraps every query in FTS5 phrase
  quotes, so multi-token queries require a contiguous token run rather than
  AND-of-terms (the quoting itself is load-bearing injection safety and
  must be kept, just not as a whole-query phrase).
- **TASK-3996** — the engine's keyword leg only joins `Media`/`media_fts`,
  so notes and conversations are structurally unreachable through hybrid
  search's FTS leg regardless of query content.
- **TASK-3997** — investigation: the four-seam (plain) keyword path
  AND-joins every query term group, so one term with no match anywhere in
  the corpus zeroes the whole query's result set. Filed as an
  investigation/product-judgment task, not a defect with an obvious fix.

Do not "fix" any of these by editing the harness or the fixtures — the
harness's job is to keep measuring the real seam accurately; the numbers it
currently reports for hybrid mode *are* today's truth about that seam.

## Adding a fixture document or golden query

1. Add a `[[doc]]` entry to `fixtures/corpus.toml` (`slug`, `source_type`
   ∈ `note`/`media`/`conversation`, `title`, `content`) or a `[[query]]`
   entry to `fixtures/golden.toml` (`id`, `query`, `category`,
   `relevant_slugs` — a list of corpus slugs, `[]` only for `category =
   "negative"`).
2. Run `pytest Tests/RAG_Eval/test_goldenset_integrity.py -q` — the
   validator fails fast and lists **every** structural defect at once
   (duplicate slugs/ids, an unknown category or source_type, a
   `relevant_slugs` entry with no matching document, a category or source
   type with zero members). Fix everything it reports before moving on.
3. If you touched `corpus.toml` or `golden.toml`, the fixture SHA-256 in
   `current_fingerprint()` changes, which means the *next* gated run will
   report `ENVIRONMENT_CHANGED` against the committed baselines rather than
   scoring anything — re-stamp deliberately (see above), with the new
   fixture content named as the reason in the PR, not silently absorbed
   into an unrelated change's re-stamp.
