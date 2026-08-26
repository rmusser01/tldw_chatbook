# RAG Server-Port P1 Eval Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A pytest-only, env-gated retrieval eval harness: fixture corpus + golden set run through the Library seam under three profile modes, scored with ported P/R/MRR/NDCG metrics against committed, environment-fingerprinted baselines with fail-on-regression gating.

**Architecture:** Three pure ported modules (`metrics`/`regression`/`gating`) under `tldw_chatbook/RAG_Search/eval/`; data + harness under `Tests/RAG_Eval/` (fixtures, loader, runner, baselines). The harness ingests the corpus through the REAL writer APIs into scratch DBs, indexes through the real document builders into its OWN `create_rag_service` instance, and queries through `LibraryLocalRagSearchService`. Doc-level metrics via provenance canonicalization.

**Tech Stack:** Python 3.11+, pytest, pydantic v2, loguru, sentence-transformers/chromadb (via existing `embeddings_rag` extra), tomllib.

## Global Constraints

- Spec (authority): `Docs/superpowers/specs/2026-08-08-rag-port-p1-eval-harness-design.md` — read before any task.
- Worktree `.worktrees/rag-port-p1`, branch `feat/rag-port-p1-eval-harness`. **cwd silently resets between Bash blocks — start EVERY block with `cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/rag-port-p1`.**
- Tests: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <paths>` with cwd in the worktree. pytest is the ONLY python entry point (never `python -c` app imports). "no tests ran" = FAILED gate; read numeric counts.
- Env gates (exact names): `RAG_EVAL=1` enables the harness tests (otherwise they SKIP with reason "set RAG_EVAL=1 to run the retrieval eval harness"); `RAG_EVAL_UPDATE_BASELINES=1` re-stamps baselines (printing per-metric old→new deltas).
- The harness NEVER calls `get_shared_rag_service` (TASK-408 leak) and never touches live config/DBs — `Tests/conftest.py::isolate_test_environment` (autouse) already isolates HOME/XDG/`TLDW_CONFIG_PATH`; do not disable it.
- The engine's media keyword leg reads `config.search.media_db_path` — the harness sets it to its scratch media DB (exercises P0's validated injection point).
- Ported code carries a module-docstring attribution line: "Ported from tldw_server2 rag_service/<file> (P1, RAG server-port programme)."
- Never `git stash`; Edit-based restores; targeted test runs; push after every task; commits end `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` and reference the P1 backlog task ID (assigned in Task 1).

## Verified code anchors (line numbers drift; grep first)

- Writer APIs: `MediaDatabase.add_media_with_keywords(title=, media_type=, content=)` → `(media_id, uuid, message)`; `CharactersRAGDB.add_conversation({"title": ...})` → conv_id; `.add_message({"conversation_id", "sender", "content", "timestamp"})`; `.add_note(...)` (read its signature at DB/ChaChaNotes_DB.py ~L10444).
- App-shape recipe (copy from `Tests/Library/test_library_local_rag_search_service.py` ~L180-215): `SimpleNamespace(media_reading_scope_service=MediaReadingScopeService(LocalMediaReadingService(media_db), None), chachanotes_db=conversations, notes_scope_service=NotesScopeService(...), prompt_scope_service=None, _rag_service=<harness service>)`. An unstamped `_rag_service` injection wins outright in `_resolve_rag_runtime` (deliberate P0 carve-out) — exactly what the harness wants.
- Ingestion metadata contract (`RAG_Search/ingestion_indexing.py` ~L545/590/635): every indexed doc carries `source_id` (str DB id), `title`, `source_type` (`ITEM_TYPE_MEDIA`/`_NOTE`/`_CONVERSATION`); document builders `media_document`/`note_document`/`conversation_document` exist there — index via `service.index_document(...)` per built doc, or the module's `index_entries` helper (read both, pick the narrower seam that doesn't need the daemon thread).
- `SearchConfig` (`RAG_Search/simplified/config.py` ~L298) is an unfrozen dataclass → per-mode switching = assign `service.config.search.default_search_mode` between passes ("plain"/"semantic"/"hybrid"). One service, one Chroma persist dir (Chroma refuses a second persistent client on one path in-process).
- Seam: `LibraryLocalRagSearchService(app).search(query=..., source_types=("notes","media","conversations"), mode="rag", top_k=K, ...)` — read the real signature (~L126) before writing the runner; rows come back on the outcome object (read `Library/library_rag_service.py::run_library_rag_search` vs calling the service directly — the harness calls the SERVICE directly; `run_library_rag_search` needs more app-shape).
- Server sources to port (read-only): `/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/core/RAG/rag_service/{retrieval_metrics.py,regression.py,quality_gating.py}` (263/418/304 lines; stdlib+loguru+pydantic only).
- Pydantic pin `>=2.4,<3` — server models use v2 idioms (`model_config = {"frozen": True}`) — compatible.

---

### Task 1: Backlog filing

**Files:** Create: `backlog/tasks/task-<ID> - RAG-port-P1-retrieval-eval-harness.md` (via CLI)

**Interfaces:** Produces the P1 backlog ID used in every later commit.

- [ ] **Step 1:** Scan max task ID across ALL worktrees + origin/dev (the CLI auto-assigns from the LOCAL max — unsafe; ten+ historical collisions):

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/rag-port-p1
for d in /Users/macbook-dev/Documents/GitHub/tldw_chatbook /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/*; do ls "$d/backlog/tasks" 2>/dev/null; done | grep -oE '^task-[0-9]+' | grep -oE '[0-9]+' | sort -n | tail -2
git fetch origin dev --quiet && git ls-tree -r --name-only origin/dev -- backlog/tasks | grep -oE 'task-[0-9]+' | grep -oE '[0-9]+' | sort -n | tail -2
```

Leapfrog max+100. If the CLI assigns a lower ID: create → `mv` the file → patch frontmatter `id:` → verify with `backlog task <ID> --plain` (recipe proven in P0).

- [ ] **Step 2:** Create with `-s "In Progress"`, `--plan` referencing spec+plan paths, and one `--ac` per spec outcome (repeat the flag): metrics module with known-answer tests; regression/gating with fingerprint outcome; corpus spans notes+media+conversations; golden set slug-keyed with per-category validator; harness env-gated, Library-seam, three modes, doc-level canonicalized metrics; committed fingerprinted baselines with reviewable update deltas; fail-on-regression gate; negative-category split measures; per-mode latency reported ungated.
- [ ] **Step 3:** Commit `chore(backlog): file RAG-port P1 eval-harness task` + trailer; push.

---

### Task 2: Ported metrics module (always-on tests)

**Files:**
- Create: `tldw_chatbook/RAG_Search/eval/__init__.py`, `tldw_chatbook/RAG_Search/eval/metrics.py`
- Test: `Tests/RAG_Eval/test_metrics.py` (create; NO env gate — pure functions run in routine suites)

**Interfaces:**
- Produces: `precision_at_k(retrieved_ids, relevant_ids, k) -> float`, `recall_at_k`, `mrr`, `ndcg_at_k`, `f1_at_k`, `@dataclass RetrievalMetrics(precision, recall, mrr, ndcg, f1, k)`, `evaluate_retrieval(retrieved_ids, relevant_ids, k=10) -> RetrievalMetrics`, `evaluate_retrieval_batch(results: list[tuple[list[str], list[str]]], k=10) -> dict[str, float]` — signatures identical to the server module.

- [ ] **Step 1: Write the failing known-answer tests** (hand-computed values; these are the oracles the whole harness rests on):

```python
# Tests/RAG_Eval/test_metrics.py
"""Known-answer tests for the ported retrieval metrics (always-on; pure)."""
import math
import pytest
from tldw_chatbook.RAG_Search.eval.metrics import (
    evaluate_retrieval, evaluate_retrieval_batch, f1_at_k, mrr,
    ndcg_at_k, precision_at_k, recall_at_k,
)


def test_precision_at_k_hand_computed():
    # 2 of top-3 relevant
    assert precision_at_k(["a", "x", "b"], ["a", "b", "c"], k=3) == pytest.approx(2 / 3)

def test_recall_at_k_hand_computed():
    # 2 of 3 relevant found in top-3
    assert recall_at_k(["a", "x", "b"], ["a", "b", "c"], k=3) == pytest.approx(2 / 3)

def test_mrr_first_relevant_at_rank_2():
    assert mrr(["x", "a", "b"], ["a", "b"]) == pytest.approx(0.5)

def test_ndcg_at_k_hand_computed():
    # relevant at ranks 1 and 3 of k=3, 2 relevant total:
    # DCG = 1/log2(2) + 0 + 1/log2(4) = 1 + 0.5 = 1.5
    # IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.63093
    expected = 1.5 / (1 + 1 / math.log2(3))
    assert ndcg_at_k(["a", "x", "b"], ["a", "b"], k=3) == pytest.approx(expected)

def test_f1_is_harmonic_mean_of_p_and_r():
    p = precision_at_k(["a", "x"], ["a", "b", "c"], k=2)   # 0.5
    r = recall_at_k(["a", "x"], ["a", "b", "c"], k=2)      # 1/3
    assert f1_at_k(["a", "x"], ["a", "b", "c"], k=2) == pytest.approx(2 * p * r / (p + r))

def test_boundaries_empty_retrieved_and_empty_relevant():
    assert precision_at_k([], ["a"], k=5) == 0.0
    assert recall_at_k([], ["a"], k=5) == 0.0
    assert mrr([], ["a"]) == 0.0
    # Empty relevant: whatever convention the server module uses, pin it —
    # read the ported bodies and assert the exact behavior (0.0 expected).
    assert recall_at_k(["a"], [], k=5) == 0.0

def test_evaluate_retrieval_k_below_one_raises():
    with pytest.raises(ValueError):
        evaluate_retrieval(["a"], ["a"], k=0)

def test_batch_averages_across_queries():
    batch = [(["a"], ["a"]), (["x"], ["a"])]   # P@1 = 1.0 and 0.0
    out = evaluate_retrieval_batch(batch, k=1)
    assert out["precision"] == pytest.approx(0.5)
```

- [ ] **Step 2: Run → FAIL** (module missing). **Step 3:** Port `retrieval_metrics.py` near-verbatim (attribution line; keep signatures/docstrings; drop nothing). **Step 4: Run → all pass; read the count.** **Step 5:** Commit `feat(rag-eval): port retrieval metrics (P/R/MRR/NDCG/F1@k) with known-answer tests`; push.

---

### Task 3: Ported regression + gating modules (always-on tests)

**Files:**
- Create: `tldw_chatbook/RAG_Search/eval/gating.py`, `tldw_chatbook/RAG_Search/eval/regression.py` (this import order: regression imports gating, as on the server)
- Test: `Tests/RAG_Eval/test_regression_gating.py` (create; always-on)

**Interfaces:**
- Produces: `MetricBaseline` (frozen pydantic: `baseline_id, created_at, pipeline_config, metrics, metadata`), `RegressionDetector(baseline_dir: Path)` with `save_baseline`/`load_baseline`/`detect(current_metrics, baseline_id, config: GatingConfig) -> RegressionReport`; `GatingConfig`, `GatingEvaluator`, `GatingResult` (pass/warn/fail enum); plus one chatbook-side helper in `regression.py`:
  `environment_mismatch(baseline: MetricBaseline, fingerprint: dict) -> list[str]` — returns the list of differing fingerprint keys (empty = match). Task 7 builds the fingerprint dict; this helper only compares `baseline.metadata.get("environment", {})` vs the given dict.

- [ ] **Step 1: Failing tests**: baseline JSON round-trip (`save_baseline` → `load_baseline` equality); a planted regression (baseline precision 0.8, current 0.5, fail band) → `RegressionReport` flags fail; a within-warn-band dip → warn not fail; `environment_mismatch` returns `["sentence_transformers"]` when only that key differs and `[]` on match. Write them against the exact server class APIs (read the server file first; copy field names precisely).
- [ ] **Step 2: RED. Step 3:** Port both modules verbatim + add `environment_mismatch` (Google docstring). **Step 4: GREEN; counts. Step 5:** Commit `feat(rag-eval): port regression baselines + quality gating; environment_mismatch helper`; push.

---

### Task 4: Fixture corpus + golden set + integrity validator

**Files:**
- Create: `Tests/RAG_Eval/fixtures/corpus.toml`, `Tests/RAG_Eval/fixtures/golden.toml`, `Tests/RAG_Eval/harness/__init__.py`, `Tests/RAG_Eval/harness/goldenset.py`
- Test: `Tests/RAG_Eval/test_goldenset_integrity.py` (always-on)

**Interfaces:**
- Produces: `corpus.toml` — `[[doc]]` entries: `slug` (unique str), `source_type` ("note"|"media"|"conversation"), `title`, `content`. `golden.toml` — `[[query]]` entries: `id`, `query`, `category` ("keyword"|"paraphrase"|"vocabulary_mismatch"|"negative"), `relevant_slugs` (list; empty allowed ONLY for negative). `goldenset.py`: `load_corpus(path) -> list[CorpusDoc]`, `load_golden(path) -> list[GoldenQuery]` (frozen dataclasses mirroring the TOML fields), `validate(corpus, golden) -> None` (raises `GoldenSetError` naming every defect: duplicate slugs, unknown `relevant_slugs`, empty relevant on non-negative, empty category, empty corpus source type).

- [ ] **Step 1: Author the corpus** (~45 docs: ≥12 notes, ≥18 media, ≥12 conversations, plus distractors in each type). Design rules (from the spec): planted unambiguous targets with clear relevance margins; four planned capability groups —
  - keyword-exact: rare literal tokens ("the Zephyr-9 flywheel assembly");
  - paraphrase: relevant doc says "annual revenue grew forty percent", query will say "yearly sales increased" (no token overlap);
  - vocabulary-mismatch: doc uses only a domain synonym ("myocardial infarction") for a query term ("heart attack") — expansion-advantage, dormant until P2;
  - distractors sharing surface tokens with queries but off-topic.
  Every doc ≥3 sentences so chunking is non-trivial; content deterministic (no dates/randomness).
- [ ] **Step 2: Author the golden set** (~40 queries: ≥10 keyword, ≥10 paraphrase, ≥8 vocabulary_mismatch, ≥6 negative). Each non-negative query's `relevant_slugs` unambiguous by construction.
- [ ] **Step 3: Failing integrity tests** — validator passes on the real fixtures; RED-able on planted defects (duplicate slug, unknown relevant_slug, non-negative with empty relevant, category with zero queries) using inline-constructed bad data (never mutate the real fixtures in tests).
- [ ] **Step 4: Implement `goldenset.py`** (tomllib; frozen dataclasses; `GoldenSetError` message lists ALL defects, not just the first). **Step 5: GREEN; counts. Step 6:** Commit `feat(rag-eval): fixture corpus, golden set, integrity validator`; push.

---

### Task 5: Harness ingestion — corpus → real DBs → indexed service (env-gated smoke)

**Files:**
- Create: `Tests/RAG_Eval/harness/ingest.py`, `Tests/RAG_Eval/conftest.py`
- Test: `Tests/RAG_Eval/test_harness_smoke.py` (env-gated)

**Interfaces:**
- Consumes: Task 4's `load_corpus`.
- Produces:
  - `Tests/RAG_Eval/conftest.py`: `rag_eval_enabled` autouse-for-this-dir fixture that `pytest.skip("set RAG_EVAL=1 to run the retrieval eval harness")` unless env set, EXCEPT for the always-on test modules (scope the skip via a marker or filename check — simplest: the env-gated test files call a shared `require_rag_eval()` helper at module top via `pytestmark`; always-on files don't);
  - `build_eval_runtime(corpus, tmp_path) -> EvalRuntime` where `EvalRuntime` is a dataclass: `.app` (SimpleNamespace app-shape), `.service` (the harness's own RAG service), `.slug_to_source: dict[slug, tuple[source_type, source_id_str]]`, `.close()`.
  - Behavior: creates scratch `MediaDatabase` + `CharactersRAGDB` files under tmp_path; writes each corpus doc through the REAL writer API for its type; builds the service via `create_rag_service` with an explicit config (scratch `persist_directory`, `config.search.media_db_path = <scratch media db>`, default embedding model); indexes every doc through the real document builders (`media_document`/`note_document`/`conversation_document` from `ingestion_indexing` — NOT the daemon-thread queue); records slug→(source_type, source_id).
  - Skip conditions (in conftest, before any heavy import): missing `embeddings_rag` extras → skip naming `pip install "tldw_chatbook[embeddings_rag]"`; embedding model not locally cached → skip naming the model (check the HF cache path for `sentence-transformers/all-MiniLM-L6-v2` WITHOUT importing torch — `huggingface_hub.try_to_load_from_cache` or a cache-dir glob; read how `Embeddings_Lib` resolves the model first).

- [ ] **Step 1: Failing smoke test** (env-gated):

```python
# Tests/RAG_Eval/test_harness_smoke.py
import os, pytest
pytestmark = pytest.mark.skipif(
    os.environ.get("RAG_EVAL") != "1",
    reason="set RAG_EVAL=1 to run the retrieval eval harness",
)

def test_corpus_ingests_and_semantic_search_finds_a_planted_doc(tmp_path):
    from Tests.RAG_Eval.harness.goldenset import load_corpus
    from Tests.RAG_Eval.harness.ingest import build_eval_runtime
    corpus = load_corpus(...)  # real fixture path via pathlib relative to this file
    runtime = build_eval_runtime(corpus, tmp_path)
    try:
        stats = runtime.service.vector_store.get_collection_stats()
        assert stats["count"] > 0  # read the real stats shape first
        # one planted keyword-exact doc must come back through the SEAM:
        from tldw_chatbook.Library.library_local_rag_search_service import (
            LibraryLocalRagSearchService,
        )
        seam = LibraryLocalRagSearchService(runtime.app)
        # call the real seam search (read its signature; mode="rag") for the
        # planted "Zephyr-9" query and assert a row canonicalizing to the
        # planted slug's source id is present in the top 5
        ...
    finally:
        runtime.close()
```

  Fill `...` against the real APIs; assertions above are the contract.
- [ ] **Step 2: RED (run WITH `RAG_EVAL=1`). Also verify the skip: run WITHOUT the env var → the file reports skipped, not failed.** **Step 3:** Implement `ingest.py` + `conftest.py`. **Step 4: GREEN with RAG_EVAL=1; counts; also run the always-on Tests/RAG_Eval files WITHOUT the env var and confirm zero skips there.** **Step 5:** Commit `feat(rag-eval): env-gated harness runtime — real-API ingestion into an isolated indexed service`; push.

---

### Task 6: Harness execution — three modes, canonicalization, report

**Files:**
- Create: `Tests/RAG_Eval/harness/runner.py`, `Tests/RAG_Eval/harness/canonicalize.py`
- Test: `Tests/RAG_Eval/test_canonicalize.py` (always-on; pure), `Tests/RAG_Eval/test_harness_run.py` (env-gated)

**Interfaces:**
- Consumes: `EvalRuntime` (Task 5), golden set (Task 4), `evaluate_retrieval_batch` (Task 2).
- Produces:
  - `canonicalize.py`: `rows_to_doc_ids(rows, slug_lookup: dict[tuple[str, str], str]) -> list[str]` — maps seam rows to fixture slugs via provenance `(canonical source_type, source_id)`, dedups repeats keeping first-hit rank, drops rows that map to no slug (they count as retrieved non-relevant only if kept — DECISION: keep them as synthetic ids `"unknown:<source_type>:<id>"` so junk retrieval still costs precision). Pin the exact provenance keys by reading `LibraryRagResultRow`/row provenance for all three source types FIRST; write the key names into the module docstring.
  - `runner.py`: `run_eval(runtime, golden, k=10) -> EvalReport` — for each mode in ("semantic", "plain", "hybrid"): set `runtime.service.config.search.default_search_mode = mode`, run every golden query through the seam (top_k=k), canonicalize, collect per-query `(retrieved_doc_ids, relevant_slugs)`; compute per-category and overall metrics via `evaluate_retrieval_batch`; negative queries EXCLUDED from averages — for them record keyword-mode results-returned@k and semantic/hybrid top vector similarity; record per-query wall latency, aggregate mean/p95 per mode. `EvalReport.to_dict()` and `format_summary() -> str` (the printed table, including the four-seam-keyword vs hybrid delta line). A query raising at the seam → recorded as that query's error in the report; the run continues.
- [ ] **Step 1: Always-on failing tests for `canonicalize.py`** with hand-built row dicts for all three source types + a duplicate-chunk case + an unknown row case (assert the synthetic id). **Step 2: RED → implement → GREEN.**
- [ ] **Step 3: Env-gated failing test for `run_eval`**: full run over the real fixtures; assert the report has all three modes, every non-negative category present per mode, keyword-exact category P@10 > 0 in plain AND hybrid modes (the planted literal tokens must be found — this is the harness proving retrieval works at all), latencies recorded, zero query errors. **Step 4: RED → implement → GREEN with RAG_EVAL=1; counts.** **Step 5:** Commit `feat(rag-eval): three-mode Library-seam runner with doc-level canonicalization and report`; push.

---

### Task 7: Baselines — fingerprint, update flow, regression gate

**Files:**
- Create: `Tests/RAG_Eval/harness/baseline_io.py`, `Tests/RAG_Eval/baselines/` (three committed JSONs)
- Modify: `Tests/RAG_Eval/test_harness_run.py` (add the gate test)
- Test: `Tests/RAG_Eval/test_baseline_io.py` (always-on for pure parts)

**Interfaces:**
- Consumes: `MetricBaseline`/`RegressionDetector`/`GatingConfig`/`environment_mismatch` (Task 3), `EvalReport` (Task 6).
- Produces: `baseline_io.py`:
  - `current_fingerprint(corpus_path) -> dict` — `{"model": <embedding model id>, "sentence_transformers": <version>, "corpus_sha256": <hash of corpus.toml+golden.toml bytes>, "platform": sys.platform}`;
  - `compare_or_update(report, baselines_dir, update: bool) -> GateOutcome` where `GateOutcome` is an enum-ish dataclass: `passed` / `regressed(details)` / `environment_changed(diff_keys)` / `baselines_written(deltas)`. Update mode prints every metric old→new; compare mode with fingerprint mismatch returns `environment_changed` (NOT a failure); compare with match runs `RegressionDetector` + `GatingConfig` (fail band: absolute drop > 0.05 on any averaged metric; warn: > 0.02 — write these as named constants).
- [ ] **Step 1: Always-on failing tests** for `current_fingerprint` (stable across two calls; changes when corpus bytes change — use tmp files) and `compare_or_update` with hand-built reports/baselines: regression → `regressed`; mismatched fingerprint → `environment_changed`; update → files written + deltas returned. **Step 2: RED → implement → GREEN.**
- [ ] **Step 3: The gate test** (env-gated, appended to test_harness_run.py): run `run_eval`, then `compare_or_update(update=False)`; assert the outcome is `passed` (against the baselines committed in Step 4) — this test IS the fail-on-regression gate (a genuine regression makes it fail).
- [ ] **Step 4: Produce the initial baselines**: `RAG_EVAL=1 RAG_EVAL_UPDATE_BASELINES=1` run; eyeball the printed metrics for sanity (keyword category near-perfect in plain/hybrid; paraphrase better in semantic/hybrid than plain; vocabulary_mismatch LOW everywhere — that's the planted P2 headroom; negative top-similarities below 0.5). Paste the summary table into the task report. Commit the three JSONs.
- [ ] **Step 5: Full env-gated run GREEN end-to-end; counts. Step 6:** Commit `feat(rag-eval): fingerprinted committed baselines + fail-on-regression gate`; push.

---

### Task 8: Docs + backlog closure

**Files:**
- Create: `Tests/RAG_Eval/README.md`
- Modify: the P1 backlog task (Done), `backlog/docs/lessons-testing-evidence.md` ONLY if something genuinely generalizable surfaced.

- [ ] **Step 1:** README: what the harness measures, exact invocations (run / update-baselines), the P2 discipline (before/after numbers in the PR; deliberate re-stamping only), fingerprint semantics ("environment changed" ≠ regression), category meanings incl. vocabulary_mismatch = planted P2 headroom, and the doc-level canonicalization contract.
- [ ] **Step 2:** Tick all P1 ACs, add Implementation Notes (approach, the baseline numbers table, decisions), `backlog task edit <ID> -s Done`.
- [ ] **Step 3:** Commit `docs(rag-eval): harness README; close P1 backlog task`; push.

---

### Task 9: Acceptance gates

- [ ] **Step 1:** Always-on battery: every Tests/RAG_Eval file WITHOUT `RAG_EVAL` (pure tests pass, gated tests skip with the exact reason string); read counts.
- [ ] **Step 2:** Full harness: `RAG_EVAL=1` over Tests/RAG_Eval — all pass incl. the gate test against committed baselines; read counts; paste the summary table.
- [ ] **Step 3:** `--collect-only -q` over Tests/ at HEAD vs a baseline worktree at the merge-base (create `.worktrees/p1-baseline` off the merge-base, remove after): HEAD total = baseline + exactly the new test count (show arithmetic).
- [ ] **Step 4:** Fix-forward anything found (RED-first, own commits); push.

---

## Self-review (done at plan time)

- **Spec coverage:** ported modules → Tasks 2-3; corpus/golden/validator → Task 4; harness isolation/ingestion realism/injection point → Task 5; seam execution, three modes one service, canonicalization, negative split-measures, latency, report → Task 6; fingerprinted baselines, reviewable deltas, environment-changed outcome, gate semantics → Task 7; error handling → Tasks 5 (skips) + 6 (per-query errors) + 4 (validator); harness's own always-on tests → Tasks 2/3/4/6/7; out-of-scope respected (no answer-layer, no MCP, no tuning).
- **Placeholder scan:** the `...` in Tasks 5/6 test skeletons are explicit fill-against-real-API instructions with stated contracts — per this repo's proven P0 pattern for codebase-integration steps; no TBD/TODO remain.
- **Type consistency:** `EvalRuntime` (Task 5) consumed by Task 6/7 runners; `EvalReport` (Task 6) consumed by Task 7's `compare_or_update`; `environment_mismatch` defined Task 3, used Task 7; slug mapping `slug_to_source` (Task 5) inverted into `slug_lookup` for Task 6's canonicalizer — the inversion happens in Task 6's runner, documented there.
- **Known risk pushed to execution:** exact seam-call signature and row provenance keys are read-first steps with contract assertions; Chroma stats dict shape read-first in Task 5.
