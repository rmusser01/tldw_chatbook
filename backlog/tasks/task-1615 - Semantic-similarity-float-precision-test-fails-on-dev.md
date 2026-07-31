---
id: TASK-1615
title: >-
  Semantic-similarity float-precision test fails on dev
status: To Do
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
- [ ] The test passes deterministically, with the fix in either the fixture or the short-circuit sentinel
<!-- AC:END -->
