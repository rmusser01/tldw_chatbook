---
id: TASK-16327
title: Add a research report self-eval harness
status: Done
assignee:
  - '@robert'
created_date: '2026-08-15 05:16'
updated_date: '2026-08-15 16:10'
labels:
  - research
  - evals
dependencies:
  - TASK-16331
  - TASK-16322
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
There is no way to measure whether pipeline changes improve report quality. mole ships self-evaluation with grounding rates; the chatbook already has an Evals module. Add a small eval runner that scores research reports on citation accuracy and grounding using the verification data produced by the citation verification work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An eval runner scores reports on citation accuracy and grounding using verification outcomes from the pipeline
- [x] #2 The runner integrates with the existing Evals framework rather than a parallel harness
- [x] #3 A baseline metric set is recorded for the current pipeline
- [x] #4 Tests cover the scoring logic with synthetic verification payloads
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Survey the Evals framework (eval_orchestrator, eval_runner, task-specific runners) for the integration pattern
2. TDD a research-report scorer: citation accuracy and grounding metrics computed from stored verification payloads (claims.json + verification_summary.json)
3. Register a research eval task/runner in the existing framework with a recorded baseline metric set for the current pipeline
4. Tests with synthetic verification payloads plus lint plus task close
ADR required: no - read-only scoring over existing artifacts, registered through the existing Evals extension point
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- New `Evals/research_report_scorer.py`: `score_research_report(verification)` computes four deterministic metrics in [0,1] from the pipeline's existing verification payload — `citation_accuracy` (resolved/total `[n]` markers), `quote_grounding` (verbatim-verified/checked quotes), `claim_support_rate` (supported/claims, falling back to marker accuracy without per-claim detail), `cited_sentence_ratio` (cited/all sentences). `BASELINE_VERIFICATION_PAYLOAD` + `BASELINE_METRICS` pin the definitions.
- `ResearchReportRunner(BaseEvalRunner)` lives in `specialized_runners.py` alongside the other specialized runners and is dispatched from `EvalRunner`'s existing category/task-type wiring (`category == "research"` or `task_type == "research_report"`) — no parallel harness. `run_sample` scores `sample.metadata["verification"]` (or JSON in `input_text`), consults no LLM, and returns metrics plus raw counts in metadata for aggregation.
- Baseline recorded in `Docs/Development/research-report-eval-baseline.md`: metric definitions, the synthetic baseline values (0.80 / 0.75 / 0.75 / 0.625) that pin the scorer, and the procedure for recording a live baseline from completed runs' `verification_summary.json` payloads (requires configured LLMs + network).
- Verified TDD: 6 tests written first and watched failing (full-payload metrics, zero-marker edges, claim fallback, runner scoring, EvalRunner dispatch, baseline reproducibility); full `Tests/Evals/` = 599 passed, 12 skipped (no regressions); ruff clean.
<!-- SECTION:NOTES:END -->
