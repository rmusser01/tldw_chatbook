---
id: TASK-3894
title: RAG-port P1 retrieval eval harness
status: In Progress
assignee: []
created_date: '2026-08-09 02:18'
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
- [ ] #1 Metrics module (precision/recall/MRR/NDCG/F1@k) ships with known-answer unit tests: hand-computed values on tiny lists plus boundary cases (k=1, empty retrieved, empty relevant).
- [ ] #2 Regression and gating modules round-trip a baseline JSON, detect a planted regression, respect configured warn/fail bands, and have a test covering the fingerprint-mismatch outcome.
- [ ] #3 Fixture corpus spans at least notes, media, and conversations so the four-seam keyword mode is exercised and measured, not just semantic search.
- [ ] #4 Golden set is a single TOML file keyed by stable fixture slugs (never DB ids), each entry carrying a category (keyword / paraphrase / vocabulary-mismatch / negative), validated by an integrity check that fails fast on a malformed fixture (duplicate ids, a relevant_ids entry with no matching corpus doc, an empty category).
- [ ] #5 Harness is opt-in via an env gate (skips by default, runs under RAG_EVAL=1), runs every golden query through the Library retrieval seam (LibraryLocalRagSearchService) across all three profile modes -- semantic, plain four-seam keyword, hybrid -- on one indexed service per run, and canonicalizes chunk-level rows to document level via source-type/source-id provenance before computing metrics, deduping duplicate chunks of one document to the first-hit rank.
- [ ] #6 Per-mode metric baselines are committed as JSON files carrying an environment fingerprint (model, library versions, corpus content hash, platform); the baseline-update run prints every metric's old-to-new delta so the baseline commit's diff is reviewable, never a silent overwrite.
- [ ] #7 With matching fingerprints, a fail-level regression against the baseline makes the harness pytest run fail (nonzero exit); the run never fails on a fingerprint mismatch, which instead reports environment-changed-re-baseline as a distinct outcome.
- [ ] #8 Negative-category queries are excluded from the averaged precision/recall/MRR/NDCG and instead report their own per-mode measure: results-returned@k for keyword mode, top vector similarity for semantic/hybrid.
- [ ] #9 Per-mode latency (mean/p95) is reported in the harness output but never gates the run (pass/fail depends only on the metric regression check).
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Spec: Docs/superpowers/specs/2026-08-08-rag-port-p1-eval-harness-design.md. Implementation plan: Docs/superpowers/plans/2026-08-08-rag-port-p1-eval-harness.md. This task (Task 1 of the plan) files the backlog task; the plan's Tasks 2-9 (ported metrics module, regression+gating modules, corpus+golden set+validator, harness ingestion runtime, three-mode runner+canonicalization, baselines+regression gate, docs+backlog closure, acceptance gates) carry out the design and land in this same backlog task's Implementation Notes on completion.
<!-- SECTION:PLAN:END -->
