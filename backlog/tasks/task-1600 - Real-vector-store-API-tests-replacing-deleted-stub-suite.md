---
id: TASK-1600
title: >-
  Write real vector_store.py API tests replacing the deleted stub suite
status: To Do
assignee: []
created_date: '2026-07-30 23:20'
labels:
  - testing
  - rag
priority: medium
dependencies: [task-1464]
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-1464 deleted `Tests/RAG/simplified/test_vector_stores.py` (~900 lines): it imported `tldw_chatbook.RAG_Search.simplified.vector_stores` — plural, a module that never existed — caught the ImportError, defined placeholder classes inside the test file, and tested those stubs. Permanently green, testing nothing. The real module (`vector_store.py`, singular — InMemory/Chroma vector stores) deserves a small, real test file: construction, add/query/delete round-trips, persistence behavior where applicable, and the skip-gating for the chromadb extra. Keep it modest — the stub suite's 900 lines are not the bar; a few dozen real assertions beat them.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

- [ ] A test file imports the REAL `vector_store` module (no fallback stubs) and fails collection if the import breaks
- [ ] Round-trip coverage for the in-memory store; chromadb-backed coverage gated on the extra
- [ ] Tests pass on a machine without the chromadb extra (gated skips) and with it
