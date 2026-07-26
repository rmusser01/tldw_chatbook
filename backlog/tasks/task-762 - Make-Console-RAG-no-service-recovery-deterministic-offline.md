---
id: TASK-762
title: Make Console RAG no-service recovery deterministic offline
status: To Do
assignee: []
created_date: '2026-07-26 17:57'
labels:
  - console
  - rag
  - baseline
  - offline
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the Console Library RAG no-service path stage its recoverable blocked state without attempting embedding-model initialization or network access, eliminating the deterministic offline baseline failure inherited from dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No-service Console RAG action stages a blocked recoverable result,The no-service path performs no embedding download or network access,Existing configured-service RAG staging remains unchanged,The exact no-service regression and focused Console RAG tests pass offline
<!-- AC:END -->
