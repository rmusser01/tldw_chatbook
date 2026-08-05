---
id: TASK-863
title: >-
  Evals/metrics_calculator.py is a 506-line orphan the README tells users to
  import
status: Done
assignee: []
created_date: '2026-07-27 04:00'
updated_date: '2026-07-27 05:40'
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
- [x] #1 Exactly one `MetricsCalculator` implementation remains importable
- [x] #2 `Evals/README.md` and `DEVELOPER_GUIDE.md` point at the surviving one
- [x] #3 The surviving `calculate_semantic_similarity` clamps to `[0, 1]` and cannot return a negative score
- [x] #4 A test would fail if the two implementations were reintroduced and drifted
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Chose option (a): deleted the orphan `metrics_calculator.py` and kept `eval_runner.py`'s `MetricsCalculator` as the single source, repointing docs at it.

Why (a) over (b)/(c): `eval_runner.py`'s copy is already the one used in production (`specialized_runners.py` imports `BaseEvalRunner` etc. from `eval_runner.py` at module level regardless, so its "heavy" import - pulls in `tldw_chatbook.config`, `Chat_Functions.chat_api_call`, ~0.7s - is already paid by every real consumer; the two inline `from .eval_runner import MetricsCalculator` in `specialized_runners.py` are just local-name style, not genuine lazy-loading, since the module is already cached in sys.modules by the top-level import). No consumer benefits from the orphan's lighter weight, so re-exporting or flipping the direction (b/c) would only add risk (larger diff inside a 2600-line production file) for no realized benefit. Re-export (c) was rejected because getting the class via import means eval_runner.py loads anyway, so it doesn't preserve the "lightweight" property either - it only adds an indirection with no upside.

Orphan-unique methods (verified via AST/grep, none had any callers outside `metrics_calculator.py` itself or docs) - all three moved verbatim into `eval_runner.py`'s `MetricsCalculator`:
- `calculate_rouge_scores` - bundles rouge_1/2/l into one dict.
- `calculate_classification_metrics` - batch-level confusion-matrix precision/recall/F1/accuracy over label lists; genuinely distinct from `ClassificationRunner.calculate_metrics` in eval_runner.py (a different, per-sample scoring method), so nothing already covered it.
- `calculate_all_metrics` - name-dispatch convenience wrapper; README.md's "Extending Metrics" example already calls `super().calculate_all_metrics(...)`, so this also fixes that example.

Found but did not treat as blocking: `eval_runner.py`'s `calculate_semantic_similarity` numpy branch upcasts embeddings to `float64` before dot/norm (comment: "a quality improvement, not an exactness guarantee"); the orphan's numpy branch did not. This is a floating-point precision refinement only (sub-ULP to ~1e-6 differences), not a logic/outcome divergence, and deleting the orphan resolves it by simply keeping the more mature copy - no behavior was changed on either the exact-match short-circuit, the `[0,1]` clamp, or the zero-guard (`else 1.0` on the pure-Python branch, confirmed still correct and unchanged).

Docs updated: `Evals/README.md` (import path), `Evals/DEVELOPER_GUIDE.md` (section header + code comment), `Evals/REFACTORING_COMPLETE.md` (file-tree entry removed + a dated note explaining the later consolidation; left the historical "Files Modified" list untouched since it's an accurate record of the 2025-08-16 refactor, not a live pointer).

Tests: `Tests/Evals/test_integration.py` import repointed to `eval_runner`. `Tests/Evals/test_evaluation_metrics.py`'s `TestStandaloneMetricsCalculatorSemanticSimilarity` (which exercised the now-deleted module directly) replaced with `TestExactlyOneMetricsCalculatorImplementation::test_only_one_metrics_calculator_class_defined_in_evals_package`, an AST-based scan of every `.py` file under `tldw_chatbook/Evals/` asserting exactly one `class MetricsCalculator` exists (in `eval_runner.py`). Verified the guard actually fires: temporarily added a second `class MetricsCalculator` in a throwaway file under `Evals/`, ran the guard test, watched it fail with a clear list-mismatch assertion naming both files, then deleted the throwaway file and reran to confirm it passes again.

`Tests/Evals/` suite: 403 passed, 13 skipped both before and after (test count in `test_evaluation_metrics.py` net -2: removed 3 standalone-copy tests, added 1 guard test). `python -c "from tldw_chatbook.app import TldwCli"` succeeds.

Files changed: `tldw_chatbook/Evals/eval_runner.py` (+3 methods), `tldw_chatbook/Evals/metrics_calculator.py` (deleted), `tldw_chatbook/Evals/README.md`, `tldw_chatbook/Evals/DEVELOPER_GUIDE.md`, `tldw_chatbook/Evals/REFACTORING_COMPLETE.md`, `Tests/Evals/test_integration.py`, `Tests/Evals/test_evaluation_metrics.py`.
<!-- SECTION:NOTES:END -->

## Notes

- 2026-07-27: PR #957 (TASK-862 follow-up, addressing a Qodo review finding) brought `calculate_semantic_similarity` in both copies into semantic agreement: `metrics_calculator.py`'s copy now also has the exact-string-equality short-circuit (`predicted == expected` -> `1.0`) and clamps its result to `[0, 1]` on both the numpy and pure-Python branches, matching `eval_runner.py`'s copy. Its zero-guard (`else 1.0` denominator swap on the pure-Python branch) was already correct and was left unchanged; a `norm_product > 0` guard was added to the numpy branch, which previously had no zero-guard at all and could return `NaN` for zero-magnitude vectors. New regression tests cover this copy directly in `Tests/Evals/test_evaluation_metrics.py::TestStandaloneMetricsCalculatorSemanticSimilarity`. The two copies are no longer semantically divergent - the remaining work here is consolidating them into a single implementation (this task's actual AC), not fixing further drift.
