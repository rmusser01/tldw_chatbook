---
id: TASK-862
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

## Investigation (2026-07-27, reproduced on this machine)

**It is not a flake, and it is three defects, not one.**

Reproduced directly: `calculate_semantic_similarity("the cat sat on the mat", "a cat is on a mat")` returns `None`. So does the unrelated-text case. The guard clauses return correct floats (`1.0`, `0.0`, `0.0`) — the function returns a real number only when it *fails*.

**Why it looked intermittent.** With `sentence_transformers` importable the function returns `None`; with the import blocked it returns `1.0` via the lexical fallback. Verified both by simulating the ImportError. So it fails on a developer machine with the model cached and passes on a bare CI box. Environment-dependent, never time-dependent — "flake" was a misdiagnosis.

**Defect 1 — the result is computed and discarded on both success paths.** `eval_runner.py:774-776` (numpy) and `:782` (pure Python) are bare expression statements. Neither is returned or assigned, so control falls off the end to an implicit `None` from a function annotated `-> float`.

**Defect 2 — the pure-Python zero-guard is inverted.** `:782` reads `dot_product / ((norm1 * norm2) if norm1 * norm2 > 0 else 0.0)`. When the norms vanish it divides *by* `0.0` and raises `ZeroDivisionError`, which the broad `except Exception` then converts into the lexical fallback. The evident intent was to *return* `0.0`. Confirmed by execution. Note numpy 2.4.4 is installed here, so this branch is dead in practice — but it is wrong wherever numpy is absent, which is exactly the environment it exists for.

**Defect 3 — range mismatch.** Cosine similarity is `[-1, 1]`; callers and tests treat the result as `[0, 1]` (`assert 0 < score < 1`). Unrelated text can score negative, so the fix should clamp at 0 rather than return a raw cosine.

**Blast radius — one caller crashes outright.** `_calculate_instruction_adherence` (`eval_runner.py:1262`):
- `:1331` assigns the `None` to `content_score` and appends it to `requirement_scores`; `:1338` then does `sum(requirement_scores) / len(...)`, raising `TypeError: unsupported operand type(s) for +: 'float' and 'NoneType'`. Any instruction sample carrying an `include`/`avoid` requirement therefore crashes the scorer.
- `:1341` returns the `None` straight out, which is the current failure of `TestEvaluationMetrics::test_instruction_adherence_basic` (`assert None == 1.0`).

Seven call sites total (`eval_runner.py:1197,1201,1275,1331,1341`, `specialized_runners.py:2685`, `metrics_calculator.py:500`). The non-arithmetic ones silently persist `None` into a metrics dict where a score is expected.

**Fix shape:** return the computed value on both branches; make the zero-norm case return `0.0` instead of dividing by it; clamp the cosine to `[0, 1]`. Then assert a float is returned *with* an embedding model present — the existing tests only cover the path where the dependency is missing, which is precisely why this survived.

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] `calculate_semantic_similarity` returns the computed cosine similarity on the numpy path
- [ ] It returns the computed value on the pure-Python fallback path
- [ ] A test asserts a float is returned when an embedding model is supplied, not just when one is absent
- [ ] `Tests/Evals/test_evaluation_metrics.py::test_instruction_adherence_basic` passes deterministically
<!-- AC:END -->
