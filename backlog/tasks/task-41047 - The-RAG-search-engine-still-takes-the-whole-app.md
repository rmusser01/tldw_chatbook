---
id: TASK-41047
title: "The RAG search engine still takes the whole app"
status: To Do
created_date: 2026-11-15 02:03
labels:
- follow-up
- architecture-review-2026-10-25
- architecture
- rag
priority: low
references:
- tldw_chatbook/RAG_Search/pipeline_builder_simple.py
- tldw_chatbook/RAG_Search/pipeline_functions_simple.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-36947.3 moved the RAG scope and evidence functions into RAG_Search with explicit services, but the search engine underneath still takes the app:
- `execute_pipeline(config, app, ...)` and the `perform_*` entry points in `RAG_Search/pipeline_builder_simple.py`;
- the `pipeline_functions_simple` legs, which read `app.media_db`, `app.chachanotes_db`, `app.db_config` and `app._rag_service`.

So the RAG search path still needs a TldwCli, or a fake with the same attributes, and cannot run from the other composition roots. TASK-36947.3's report proposed this as a follow-up. Re-read on dev ecad404bd4 (2026-11-15).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `execute_pipeline` and the `perform_*` entry points take the storage slice and the RAG service instead of the app
- [ ] #2 pipeline_builder_simple and pipeline_functions_simple read no attribute of the app
- [ ] #3 The RAG tests pass without an app fake
<!-- AC:END -->
