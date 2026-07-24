---
id: TASK-549
title: Restore embedding semantic similarity return contract
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-24 20:57'
updated_date: '2026-07-24 20:57'
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
- [ ] #1 Embedding-backed semantic similarity returns a finite float
- [ ] #2 Exact normalized text returns 1.0 without loading an embedding model
- [ ] #3 Zero-norm embeddings fall back to the lexical metric instead of returning NaN
- [ ] #4 Evaluation metric and eval sentinel suites pass
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
