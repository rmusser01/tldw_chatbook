---
id: TASK-3894
title: RAG-port P1 retrieval eval harness
status: Done
assignee: []
created_date: '2026-08-09 02:18'
updated_date: '2026-08-09 14:23'
labels:
  - rag
  - eval
  - testing
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
P0 (task-3170, PR #1428, merged 2026-08-08) made the retrieval tldw_chatbook already owns reachable -- profile-driven Library search mode plus Console auto-retrieve -- but deliberately made no retrieval-quality claims. P1 builds the measuring instrument: ported precision/recall/MRR/NDCG/F1@k metrics, a regression/gating layer with fingerprinted baselines, a deterministic fixture corpus and golden query set spanning notes, media, and conversations, and an env-gated pytest harness that exercises the real Library retrieval seam across all three profile modes. With this in place, every later retrieval change in the programme (P2's query expansion, HyDE, PRF, clarification gate, granularity router; any threshold or model tuning) must show its effect as measured numbers against committed baselines, run locally -- CI is intentionally dead in this repo, so the gate is a local pytest run.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Metrics module (precision/recall/MRR/NDCG/F1@k) ships with known-answer unit tests: hand-computed values on tiny lists plus boundary cases (k=1, empty retrieved, empty relevant).
- [x] #2 Regression and gating modules round-trip a baseline JSON, detect a planted regression, respect configured warn/fail bands, and have a test covering the fingerprint-mismatch outcome.
- [x] #3 Fixture corpus spans at least notes, media, and conversations so the four-seam keyword mode is exercised and measured, not just semantic search.
- [x] #4 Golden set is a single TOML file keyed by stable fixture slugs (never DB ids), each entry carrying a category (keyword / paraphrase / vocabulary-mismatch / negative), validated by an integrity check that fails fast on a malformed fixture (duplicate ids, a relevant_ids entry with no matching corpus doc, an empty category).
- [x] #5 Harness is opt-in via an env gate (skips by default, runs under RAG_EVAL=1), runs every golden query through the Library retrieval seam (LibraryLocalRagSearchService) across all three profile modes -- semantic, plain four-seam keyword, hybrid -- on one indexed service per run, and canonicalizes chunk-level rows to document level via source-type/source-id provenance before computing metrics, deduping duplicate chunks of one document to the first-hit rank.
- [x] #6 Per-mode metric baselines are committed as JSON files carrying an environment fingerprint (model, library versions, corpus content hash, platform); the baseline-update run prints every metric's old-to-new delta so the baseline commit's diff is reviewable, never a silent overwrite.
- [x] #7 With matching fingerprints, a fail-level regression against the baseline makes the harness pytest run fail (nonzero exit); the run never fails on a fingerprint mismatch, which instead reports environment-changed-re-baseline as a distinct outcome.
- [x] #8 Negative-category queries are excluded from the averaged precision/recall/MRR/NDCG and instead report their own per-mode measure: results-returned@k for keyword mode, top vector similarity for semantic/hybrid.
- [x] #9 Per-mode latency (mean/p95) is reported in the harness output but never gates the run (pass/fail depends only on the metric regression check).
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Spec: Docs/superpowers/specs/2026-08-08-rag-port-p1-eval-harness-design.md. Implementation plan: Docs/superpowers/plans/2026-08-08-rag-port-p1-eval-harness.md. This task (Task 1 of the plan) files the backlog task; the plan's Tasks 2-9 (ported metrics module, regression+gating modules, corpus+golden set+validator, harness ingestion runtime, three-mode runner+canonicalization, baselines+regression gate, docs+backlog closure, acceptance gates) carry out the design and land in this same backlog task's Implementation Notes on completion.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
STATUS: DONE -- Task 9 acceptance pass run 2026-08-09 against HEAD 60a11668d; all three gates green, all 9 acceptance criteria independently evidenced (acceptance evidence at the end of these notes).

Approach: ported tldw_server RAG_Search/eval/ (metrics, regression, gating) into tldw_chatbook unmodified where possible; built a deterministic 48-doc fixture corpus (notes/media/conversations) plus a 44-query golden set (keyword/paraphrase/vocabulary_mismatch/negative categories) in TOML, loaded and integrity-checked by harness/goldenset.py; an ingestion runtime (harness/ingest.py) that writes fixtures through the real writer APIs and indexes them through the real batch path, never the app singleton; a canonicalization layer (harness/canonicalize.py) that collapses chunk-level seam rows to document-level fixture slugs via provenance, keeping unmapped rows as unknown:<type>:<id> rather than dropping them; a three-mode runner (harness/runner.py) that flips default_search_mode and runs all 44 queries through the real LibraryLocalRagSearchService seam per mode; and a fingerprinted baseline/gate layer (harness/baseline_io.py) that expresses an absolute pass/warn band through the ported regression detector fractional arithmetic (threshold = band / baseline) and treats a fingerprint mismatch as environment_changed, never a regression. Full narrative and numbers are in Tasks 1-7 reports under .superpowers/sdd/2026-08-08-rag-port-p1-eval-harness/.

Baseline numbers (real run, 48-doc corpus, 44 golden queries, k=10, darwin, sentence-transformers 5.4.1):

mode      P@k    R@k    MRR   NDCG    F1   docs   mean-ms  p95-ms  backend
semantic  0.117  1.000  1.000 1.000  0.208  9.1     19.8     54.0  rag-semantic
plain     0.351  0.338  0.351 0.341  0.342  0.4      3.6      5.6  local-fts
hybrid    0.117  1.000  1.000 1.000  0.208  9.1     13.2     21.5  rag-hybrid

per-category recall@10 (precision@10): keyword 1.000(0.135) / 0.833(0.867) / 1.000(0.135) for semantic/plain/hybrid; paraphrase 1.000(0.103) / 0.000(0.000) / 1.000(0.103); vocabulary_mismatch 1.000(0.105) / 0.000(0.000) / 1.000(0.105). Negatives (report-only): semantic and hybrid returned something for 7/7 (max top score 0.2387); plain returned something for 0/7.

Key decisions: (1) absolute fail/warn bands (0.05/0.02 metric points) expressed through the ported fractional check_regression via threshold=band/baseline, run twice per mode -- a fractional band was rejected because 5 percent of hybrid overall precision (0.117) is noise-tight (0.006) against 5 percent of plain keyword precision (0.867, which is 0.043); (2) only metrics (60 total: overall plus per-category P/R/MRR/NDCG/F1 across 3 modes) are gated -- latency, mean_docs_at_k, counts and negatives all live in metadata.report_only and never gate, because Task 6 measured latency swinging 1.7-2.2x on process order alone; (3) a missing baseline and a disappeared metric both FAIL the gate (a green gate that checked nothing is worse than a red one); (4) pipeline_config (k, profile, source_types) rides alongside the fingerprint so a k change reads as environment_changed rather than a regression.

Findings surfaced by the harness (not harness defects -- filed as P2 backlog follow-ups, NOT fixed in this task): hybrid search is byte-identical to semantic search on this corpus because RRF fuses on mismatched id spaces and the FTS leg contribution never survives the fused top-k (TASK-3994); the engine keyword leg wraps every query in FTS5 phrase quotes, blocking non-contiguous multi-token matches (TASK-3995); the engine keyword leg only joins Media/media_fts, so notes and conversations are unreachable via the hybrid FTS leg (TASK-3996); the four-seam plain path AND-joins every term group, zeroing a query on one absent term (TASK-3997, filed as an investigation/product-judgment task). vocabulary_mismatch also turned out to be at ceiling (1.000) in vector modes on this corpus and model -- the category currently measures the plain-versus-vector delta, not P2 query-expansion headroom; documented as a caveat in Tests/RAG_Eval/README.md rather than treated as already solved.

Files: Tests/RAG_Eval/README.md, Tests/RAG_Eval/conftest.py, Tests/RAG_Eval/test_metrics.py, test_regression_gating.py, test_goldenset_integrity.py, test_canonicalize.py, test_baseline_io.py, test_runner_error_paths.py, test_harness_smoke.py, test_harness_run.py; Tests/RAG_Eval/harness/environment.py, goldenset.py, ingest.py, canonicalize.py, runner.py, baseline_io.py; Tests/RAG_Eval/fixtures/corpus.toml, golden.toml; Tests/RAG_Eval/baselines/semantic.json, plain.json, hybrid.json. Counts: 128 passed / 4 skipped ungated; 132 passed gated in both process orderings; ruff clean.

Backlog follow-ups filed from this task findings: TASK-3994 (hybrid RRF id-space mismatch, high priority), TASK-3995 (engine keyword-leg phrase quoting, high priority), TASK-3996 (engine keyword-leg media-only, medium priority), TASK-3997 (four-seam AND-strictness, investigation, medium priority).

### Acceptance evidence (2026-08-09, HEAD 60a11668d, darwin, python 3.12 venv)

Always-on battery (`pytest Tests/RAG_Eval` with RAG_EVAL unset): **128 passed, 4 skipped** in 0.75s. All 4 skips are the env-gated tests (test_harness_run.py x2, test_harness_smoke.py x2), each with the exact reason string `set RAG_EVAL=1 to run the retrieval eval harness`. No gated test leaks into the default suite, and no always-on test needs the gate.

Full harness (`RAG_EVAL=1 pytest Tests/RAG_Eval`): **132 passed** in 9.7-10.1s. The gate test ran against the committed baselines and printed `PASSED: No regression. 60 metric(s) within 0.05 of baseline.` with **all 60 deltas at +0.000** -- the committed baselines reproduce exactly on this machine. Reported run: 44 golden queries (37 scored, 7 negative), k=10; semantic 0.117/1.000/1.000/1.000 P/R/MRR/NDCG (9.1 docs, 7.8 mean ms, 9.9 p95), plain 0.351/0.338/0.351/0.341 (0.4 docs, 3.0/4.8 ms), hybrid identical to semantic (the known TASK-3994 defect, recorded as today's truth). Latency swung against the numbers stamped in the baselines (semantic 19.8 -> 7.8 mean ms) and the gate did not care -- AC #9 demonstrated by a real divergence, not only by construction.

Gate bite, both directions, proven end-to-end by tampering committed files and restoring them byte-exact (tree verified clean after each): (a) planted regression -- semantic `overall.precision` baseline raised +0.10 -> gate `REGRESSED: 1 metric(s) fell further than 0.05 below baseline -- semantic/overall.precision`, pytest **exit 1**; (b) fingerprint mismatch -- `platform` changed in all three baselines -> `ENVIRONMENT_CHANGED: Environment changed -- re-baseline, do not read these numbers as a regression. Differing: platform`, pytest **exit 0** plus an explicit `NOTE: ... nothing was gated` line. AC #7 holds in both directions.

Collection arithmetic (`pytest Tests/ --collect-only -q`, RAG_EVAL unset, both sides): merge-base `3023578c0` in a throwaway worktree collected **33,520**; HEAD collected **33,652**; delta **+132**. A set-diff of the two node-id lists shows 132 ids added, **0 removed, and every added id under Tests/RAG_Eval/** -- no incidental collection churn anywhere else in the suite. The 132 reconcile against the source: 112 `def test_` functions, plus parametrize expansion of 13 canonicalization cases (17 funcs -> 29 items, +12) and 5 morphological pairs across 2 golden-set tests (39 funcs -> 47 items, +8): 112 + 12 + 8 = 132. Per file: baseline_io 26, canonicalize 29, goldenset_integrity 47, harness_run 2, harness_smoke 2, metrics 8, regression_gating 9, runner_error_paths 9. The branch adds only new files (23 under Tests/RAG_Eval, 4 under tldw_chatbook/RAG_Search/eval, docs/backlog) and modifies exactly one pre-existing file (`backlog/docs/lessons-testing-evidence.md`) -- no existing test was touched.

Static analysis: `ruff check Tests/RAG_Eval/ tldw_chatbook/RAG_Search/eval/` -- all checks passed. No defects were found during the acceptance pass, so no fix-forward commits were needed.
<!-- SECTION:NOTES:END -->
