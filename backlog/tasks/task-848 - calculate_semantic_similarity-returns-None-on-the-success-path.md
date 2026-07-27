---
id: TASK-848
title: >-
  calculate_semantic_similarity discards its result and returns None on the success path
status: To Do
assignee: []
created_date: '2026-07-27 01:00'
labels:
  - evals
  - bug
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while investigating a pre-existing test flake during PR 3b of the Evals rebuild. It is not a flake — it is a real defect, and it is in the legacy orchestrator path (`Evals/eval_runner.py`), not the new word bench engine.

`MetricsCalculator.calculate_semantic_similarity` is annotated `-> float`. It has five `return` statements: three guard clauses at the top, and two inside `except` handlers. **The success path has none.**

Both cosine-similarity computations are evaluated and then thrown away — the numpy branch computes `dot(pred, exp) / (norm(pred) * norm(exp))` as a bare expression statement, and the pure-Python fallback does the same with `dot_product / (norm1 * norm2)`. Neither result is returned or assigned.

So whenever sentence-transformers is actually available and the embeddings encode successfully, the function falls off the end and returns `None`. It only returns a real number when it **fails** and lands in one of the `except` handlers, which return the lexical fallback.

That inversion is what makes it hard to notice: the function looks correct in every environment where the optional dependency is missing, and silently returns `None` in exactly the environment it was written for. Any caller performing arithmetic on the result gets a `TypeError`, and any caller storing it records a null score where a similarity was expected.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] `calculate_semantic_similarity` returns the computed cosine similarity on the numpy path
- [ ] It returns the computed value on the pure-Python fallback path
- [ ] A test asserts a float is returned when an embedding model is supplied, not just when one is absent
- [ ] `Tests/Evals/test_evaluation_metrics.py::test_instruction_adherence_basic` passes deterministically
<!-- AC:END -->
