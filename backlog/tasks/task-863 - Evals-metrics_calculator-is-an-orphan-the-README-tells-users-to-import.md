---
id: TASK-863
title: >-
  Evals/metrics_calculator.py is a 506-line orphan the README tells users to import
status: To Do
assignee: []
created_date: '2026-07-27 04:00'
labels:
  - evals
  - dead-code
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while fixing TASK-862. `tldw_chatbook/Evals/metrics_calculator.py` defines a `MetricsCalculator` class that duplicates the one in `eval_runner.py`. No Python file imports it — the only references anywhere in the tree are documentation.

That duplication is the root cause of TASK-862. The two copies drifted, and `eval_runner.py`'s copy silently lost the `cosine_sim =` assignment and its `return` from `calculate_semantic_similarity`, plus had its zero-guard changed from `else 1.0` to `else 0.0` (turning a correct `0/1.0 = 0.0` into a `ZeroDivisionError`). Nothing detected the drift because nothing compares the copies.

**The good news, established by an AST diff of the two classes:** of the 11 methods they share, 9 are byte-identical, and `calculate_perplexity` differs only cosmetically (`return math.exp(x)` vs assign-then-return). Only `calculate_semantic_similarity` had genuinely drifted. So there is no second latent bug of the same class hiding here — but there is nothing preventing the next one either.

**The live hazard is the documentation.** `Evals/README.md:309` instructs `from tldw_chatbook.Evals.metrics_calculator import MetricsCalculator`, and `DEVELOPER_GUIDE.md` presents `metrics_calculator.py` as "the Metrics System". Anyone following those docs imports the orphan, which — after TASK-862 — is the copy that still returns raw cosine values. Verified: with opposite embedding vectors it returns `-1.0`, where every caller treats the result as a `[0, 1]` score.

Decide one of: delete the orphan and repoint the docs at `eval_runner.py`, or make it the single source and have `eval_runner` import from it. Do not leave two copies.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] Exactly one `MetricsCalculator` implementation remains importable
- [ ] `Evals/README.md` and `DEVELOPER_GUIDE.md` point at the surviving one
- [ ] The surviving `calculate_semantic_similarity` clamps to `[0, 1]` and cannot return a negative score
- [ ] A test would fail if the two implementations were reintroduced and drifted
<!-- AC:END -->
