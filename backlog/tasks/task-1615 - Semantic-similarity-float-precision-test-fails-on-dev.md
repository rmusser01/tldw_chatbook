---
id: TASK-1615
title: >-
  Semantic-similarity float-precision test fails on dev
status: Done
assignee: []
created_date: '2026-07-31 15:40'
labels:
  - evals
  - tests
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/Evals/test_evaluation_metrics.py::TestSemanticSimilarityExactMatchShortCircuit::test_non_identical_strings_still_use_the_embedding_path` fails on dev (`assert 1.0 != 1.0`): two non-identical strings produce embedding similarity that rounds to exactly 1.0, colliding with the exact-match short-circuit's sentinel. Confirmed pre-existing during the task-1482 program (fails identically at the pre-branch commit; file untouched by the branch). Either the test's fixture strings need more distance or the short-circuit needs a distinct signal.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] The test passes deterministically, with the fix in either the fixture or the short-circuit sentinel
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: both tests in TestSemanticSimilarityExactMatchShortCircuit proved their path by floating-point accident — the fixture vector's cosine self-similarity lands a few ULPs short of 1.0 on some BLAS/numpy builds but EXACTLY 1.0 on this machine's, making `score != 1.0` false (the reported failure) and the exact-match test's `score == 1.0` vacuous (it would pass without the short-circuit). Fix is structural: `_ConstantEmbeddingModel` counts `encode()` calls; the exact-match test asserts the model was never consulted, the near-miss test asserts it was. Score assertions retained where platform-independent (`== 1.0` by short-circuit construction; `approx(1.0)` sanity). Mutation-verified: disabling the short-circuit fails the exact-match test via encode_calls even where the score still computes 1.0.
<!-- SECTION:NOTES:END -->
