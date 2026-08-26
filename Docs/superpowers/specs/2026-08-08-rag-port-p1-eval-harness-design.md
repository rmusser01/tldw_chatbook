# RAG Server-Port Programme — P1 Eval Harness (design)

Date: 2026-08-08
Status: approved-pending-user-review
Programme: port retrieval improvements from tldw_server2 into tldw_chatbook
Phase: P1 of 5 (P0 merged 2026-08-08: PR #1428 → dev `220345610`)

## Background

P0 made the retrieval chatbook owns reachable (profile-driven Library rag
mode: hybrid RRF / plain four-seam keyword / semantic, plus Console
auto-retrieve). P0 deliberately made **no retrieval-quality claims** — it was
reachability, verified behaviorally. P1 builds the measuring instrument so
that every later retrieval change (P2's query expansion, HyDE, PRF,
clarification gate, granularity router; threshold or model tuning) must show
its effect as measured numbers against stored baselines, locally — CI in this
repo is intentionally dead, so the gate is a local run.

Source material: tldw_server2's `retrieval_metrics.py` (P/R/MRR/NDCG/F1@k,
263 lines), `regression.py` (`MetricBaseline`/`RegressionDetector`, JSON
baselines, 418 lines), `quality_gating.py` (`GatingConfig`/`GatingEvaluator`,
304 lines). All pure stdlib + loguru + pydantic — existing chatbook
dependencies. Ported with attribution, not reinvented.

## Components

### 1. Ported modules → `tldw_chatbook/RAG_Search/eval/`

- `metrics.py` — precision/recall/MRR/NDCG/F1 @k, `evaluate_retrieval`,
  `evaluate_retrieval_batch`. Near-verbatim port.
- `regression.py` — `MetricBaseline` (frozen pydantic; `pipeline_config` +
  `metadata` slots), `RegressionDetector` (JSON save/load/compare).
- `gating.py` — `GatingConfig` thresholds (pass/warn/fail per metric),
  `GatingEvaluator`.

No UI. No wiring into the app's runtime paths — this package is imported by
the harness tests only.

### 2. Fixture corpus + golden set → `Tests/RAG_Eval/fixtures/`

- **Corpus**: ~40-60 small deterministic documents **spanning source types**
  (notes, media, conversations at minimum — required so the four-seam
  keyword mode is measured and the hybrid keyword-leg gap is quantified).
  Content plants unambiguous retrieval targets with clear relevance margins
  (metrics are rank-based; near-ties would make baselines jittery by
  construction):
  - keyword-exact hits (FTS-advantage),
  - paraphrase-only hits (vector-advantage),
  - vocabulary-mismatch pairs (expansion-advantage — dormant until P2, the
    harness proves they're NOT retrieved today),
  - distractors.
- **Golden set**: ~40 hand-authored queries in one TOML file:
  `query`, `relevant_ids` (**stable fixture slugs**, never DB ids — the
  real writer APIs assign autoincrement ids per run; the ingestion step
  records the slug→runtime-id mapping and hands it to the metric layer),
  `category` (keyword / paraphrase / vocabulary-mismatch / negative).
  Categories make P2 deltas legible per capability
  ("HyDE: paraphrase recall +0.12, keyword unchanged").
  - `negative` queries have empty `relevant_ids` (the validator permits
    this for that category only); they are excluded from averaged
    P/R/MRR/NDCG. Their per-mode measure differs because vector search
    always returns the k nearest regardless of relevance: **keyword mode
    reports results-returned@k** (FTS genuinely returns nothing on no
    match), while **semantic/hybrid report the top vector similarity**
    (expectation: below the strong-band threshold — the no-false-confidence
    measure, which later feeds P3's abstention work).
  - No `scoped` category in P1: measuring scoped-retrieval quality needs
    scope machinery in the harness that P0's behavioral tests already
    cover; it joins the harness in P2 when scope-aware hybrid lands.

### 3. Harness → `Tests/RAG_Eval/`

- **pytest is the only entry point** (standing rule — no CLI runner; a
  runner importing the app is the exact probe class that once wrote to the
  live config). Opt-in via env gate: without `RAG_EVAL=1` every harness
  test **skips** (routine collection sweeps stay clean and nothing heavy
  runs by accident). Invocation: `RAG_EVAL=1 pytest Tests/RAG_Eval/`.
- **Isolation (explicit requirements, not assumptions):**
  - standard config-isolation fixtures; the live config and live DBs are
    never touched;
  - the harness builds its **own** service via `create_rag_service` with a
    scratch persist dir and explicit config — never
    `get_shared_rag_service` (TASK-408: that process-wide singleton leaks
    across tests);
  - `config.search.media_db_path` points at the scratch media DB — the
    harness thereby exercises P0's validated injection point on every run.
- **Ingestion realism**: the corpus enters through the real writer APIs
  (`add_media_with_keywords`, the real note/conversation writers) and is
  indexed through the real document builders / `RAGService` indexing seam —
  so the harness regression-covers the metadata contract (`source_type`,
  chunk ids, titles), exactly where P0's keyword-leg and post-filter bugs
  lived. No hand-rolled loader that produces shapes the app never writes.
- **Execution seam**: every golden query runs through
  `LibraryLocalRagSearchService` (the Library seam), NOT the engine —
  P0 proved seam bugs are invisible at engine level. Each query runs under
  three modes: profile-semantic, profile-plain (four-seam keyword),
  profile-hybrid — **one service, one index, mode switched between
  passes** (Chroma commonly refuses a second persistent client on the same
  path in-process; three services are structurally ruled out).
- **Retrieved-id canonicalization**: seam rows are chunk-level; metrics are
  computed at **document level**. Each row canonicalizes to its fixture doc
  via provenance (`source_type` + `source_id`/`doc_id` — exact keys pinned
  in the plan), duplicate chunks of one doc dedup to the first-hit rank.
  Without this, a multi-chunk hit would pollute P@k with self-duplicates.
- **Embeddings**: the real default model (sentence-transformers
  MiniLM-L6-v2, local, no API cost). No mock mode — mock embeddings would
  measure nothing. Model loads once per session (~10s); full run budget
  ~1-2 minutes.
- **Report**: per-mode, per-category P@k / R@k / MRR / NDCG@k (+ overall),
  per-mode latency mean/p95 (informative, never gated), and the explicit
  four-seam-keyword vs hybrid delta (quantifies the media-only keyword-leg
  gap P2 closes). Written to a gitignored results dir; printed summary.

### 4. Baselines → `Tests/RAG_Eval/baselines/*.json` (committed)

- One baseline file per mode, produced by
  `RAG_EVAL=1 RAG_EVAL_UPDATE_BASELINES=1 pytest Tests/RAG_Eval/`. The
  update path prints every metric's old→new delta so the baseline commit's
  diff is reviewable, never a silent overwrite.
- `MetricBaseline.metadata` carries an **environment fingerprint**: model
  name, sentence-transformers version, corpus content hash, platform.
  On fingerprint mismatch the harness reports **"environment changed —
  re-baseline"** as a distinct, non-failing outcome (with the diff shown)
  instead of claiming a code regression; regression gating runs only when
  fingerprints match. Gating tolerances (warn/fail bands in
  `GatingConfig`) absorb residual float jitter.

### 5. Gate semantics

- With matching fingerprints, a `fail`-level regression vs baseline makes
  the harness test fail (nonzero pytest exit). The P2 discipline: run the
  harness before and after a retrieval change; the PR carries both numbers;
  baselines are re-stamped only deliberately (`RAG_EVAL_UPDATE_BASELINES=1`)
  with the change that justifies them.
- Never wired into routine suites, pre-commit, or app runtime.

## Out of scope (declared)

- Answer-layer scoring (abstention correctness, citation faithfulness,
  groundedness) — P3, where the graders land.
- MCP `perform_rag_search` seam evaluation — TASK-3500's MCP-only parity work first; the agent Library provider already follows the profile-driven path.
- Any retrieval behavior change. P1 measures; it does not tune. If the
  harness reveals a P0 defect, it is filed (or fixed as its own reviewed
  commit), not silently absorbed.

## Error handling

- Missing `embeddings_rag` extras → the harness skips with an explicit
  reason naming the install command (same honesty register as the app).
- Embedding model not locally cached (first-run download would be needed)
  → skip with a reason naming the model, never a failure or a surprise
  network download inside a test run.
- Corpus/golden-set integrity is validated before any query runs (ids
  unique, every `relevant_ids` entry exists in the corpus, every category
  non-empty) — a malformed golden set fails fast with the exact defect,
  never produces silently-wrong metrics.
- A query that errors at the seam is reported as that query's failure with
  the exception, and the run continues (one bad query must not hide the
  other 39 results).

## Testing (the harness's own tests — cheap, always-on)

- Metric functions: known-answer tests (hand-computed P/R/MRR/NDCG on tiny
  lists, boundary cases k=1, empty retrieved, empty relevant) — these run
  in routine suites (no env gate; pure functions).
- Regression/gating: round-trip a baseline JSON, detect a planted
  regression, respect warn/fail bands, fingerprint-mismatch path.
- Golden-set integrity validator: RED-able on a planted malformed fixture.
- The env-gated harness itself is exercised in CI-less fashion by the P1
  acceptance run: one full `RAG_EVAL=1` execution with committed baselines
  produced and checked in.

## Plan-phase verification items

1. Exact writer APIs for notes/conversations fixtures (the media API is
   known: `add_media_with_keywords`) and what minimal app-shaped object
   `LibraryLocalRagSearchService` needs (reuse Tests/Library fixtures).
2. How the harness switches profile mode per run cleanly (three service
   configs vs one service re-configured — pick the one that cannot leak
   state between modes).
3. The clean mode-switch mechanism on the single service (config field
   swap vs a per-mode config view) — whichever cannot leak state between
   passes; index once, query thrice.
3b. The exact provenance keys seam rows carry per source type for the
   doc-level canonicalization (media vs note vs conversation rows).
4. Confirm pydantic v2 API parity for the ported models in chatbook's
   pinned version.
5. Where the standing config-isolation fixture lives for non-UI tests
   (Tests/RAG_Eval must inherit it).
