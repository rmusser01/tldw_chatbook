---
id: TASK-41054
title: "Two scoped-semantic RAG tests are red on dev: the query is not embedded once per search, and note results drop out"
status: To Do
created_date: 2026-11-15 02:03
labels:
- follow-up
- architecture-review-2026-10-25
- tests
- rag
priority: low
references:
- Tests/RAG/test_scope_pipeline_enforcement.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/RAG/test_scope_pipeline_enforcement.py::TestSemanticLegSingleEmbed` fails two of its three tests on dev ecad404bd4 (2026-11-15) (serially, at `-n 4` and under the bootstrap profile):
- `test_scoped_semantic_embeds_query_once_across_allowlists` sees zero embedding calls where it expects one.
- `test_service_without_embeddings_wrapper_keeps_old_contract` returns `['m1']` where it expects `['n1', 'm1']`.

TASK-36947.3 found both red at its base on 2026-11-06. Production does not reach this leg: `pipeline_functions_simple.search_semantic` runs only under `execute_pipeline`, whose callers are the legacy entry points TASK-41048 retires and TASK-32807.2's `pipeline_loader.py`. The file runs on no PR lane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The task notes say whether the product or the tests are wrong, and name the commit
- [ ] #2 Both tests pass on dev, or are deleted with the engine under TASK-41048; a fixed product defect is shown red first
<!-- AC:END -->
