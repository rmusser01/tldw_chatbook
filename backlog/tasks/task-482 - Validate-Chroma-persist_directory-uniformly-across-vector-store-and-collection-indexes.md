---
id: TASK-482
title: >-
  Validate Chroma persist_directory uniformly across vector_store and
  collection_indexes
status: To Do
assignee: []
created_date: '2026-07-22 00:45'
updated_date: '2026-07-22 00:45'
labels:
  - rag
  - security
  - followup
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up from PR #771 (RAG SP1) review: Qodo flagged that `collection_indexes._client()` passes the config-sourced `persist_directory` straight into Chroma's `PersistentClient` without going through `Utils/path_validation.py`. This is a **pre-existing, store-wide pattern** — `ChromaVectorStore` (`vector_store.py:199/273`) already does the same — so SP1 deliberately mirrored it (validating only the new module would risk a normalized-vs-raw path-string divergence from the store → the `SharedSystemClient` per-persist_directory client-cache collision that SP1's migration explicitly avoids).

Harden it **uniformly**: validate/normalize the Chroma persist_directory once, at a shared point, so `vector_store.py` and `collection_indexes.py` always receive the identical validated path. Do not introduce a normalization difference between the two Chroma client construction sites.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] Config-sourced Chroma `persist_directory` is validated via `path_validation.py` before use.
- [ ] `vector_store.py` and `collection_indexes.py` construct their `PersistentClient` with the SAME validated path string (no divergence → no `SharedSystemClient` Settings/path collision).
- [ ] Existing RAG tests still pass; a test covers that both sites resolve to the identical validated path.
<!-- AC:END -->
