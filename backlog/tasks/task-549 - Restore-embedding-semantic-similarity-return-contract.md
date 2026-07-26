---
id: TASK-549
title: Restore embedding semantic similarity return contract
status: Done
assignee:
  - '@codex'
created_date: '2026-07-24 20:57'
updated_date: '2026-07-26 07:45'
labels:
  - evals
  - metrics
  - reliability
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore deterministic numeric results from embedding-backed semantic similarity after an automated unused-variable cleanup removed the return value.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Embedding-backed semantic similarity returns a finite float
- [x] #2 Exact normalized text returns 1.0 without loading an embedding model
- [x] #3 Zero-norm embeddings fall back to the lexical metric instead of returning NaN
- [x] #4 Evaluation metric and eval sentinel suites pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the instruction-adherence failure and add injected embedding-model coverage for numeric and zero-norm results.
2. Restore explicit cosine return values, add an exact-text fast path, and fail safely to the existing lexical fallback for invalid norms.
3. Run the focused metric file, eval sentinel suite, static checks, and resume the remaining full-suite gate.

ADR required: no
ADR path: N/A
Reason: This restores the established metric implementation and return type after a mechanical cleanup regression; it does not introduce a new evaluator boundary or scoring policy.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Summary: Restored deterministic float returns from embedding-backed semantic similarity and hardened invalid-vector handling.

Implementation:
- Added a normalized exact-text fast path returning 1.0 before any embedding model is loaded.
- Restored explicit cosine-similarity returns for NumPy and pure-Python paths.
- Rejects non-positive or non-finite denominators and non-finite similarity values by returning the existing lexical fallback.
- Added injected-model regressions for a finite 0.96 result, zero-norm lexical fallback, and exact-text model bypass.

Verification:
- Evaluation metrics plus execution-contract sentinel suite: 41 passed.
- Diagnostic/task sentinel harness: 2 passed.
- Final permitted full suite: 12,757 passed, 231 skipped, 240 warnings in 3h34m55s.
- Self-review: restores the established return contract without changing evaluator policy or provider boundaries.

ADR required: no
ADR path: N/A
Reason: Regression repair for the established metric implementation and return type.

Files modified:
- tldw_chatbook/Evals/eval_runner.py
- Tests/Evals/test_evaluation_metrics.py
- backlog/tasks/task-549 - Restore-embedding-semantic-similarity-return-contract.md
<!-- SECTION:NOTES:END -->
