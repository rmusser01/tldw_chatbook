---
id: TASK-10.8.2
title: 'Gate 1.6.2: Library native Search/RAG panel'
status: To Do
assignee: []
created_date: '2026-05-07 12:00'
labels:
  - product-maturity
  - phase-3-knowledge-study
  - gate-1-6-library-rag
dependencies:
  - TASK-10.8.1
documentation:
  - Docs/superpowers/specs/2026-05-06-destination-layout-ia-contracts-design.md
  - Docs/superpowers/plans/2026-05-07-gate-1-6-library-native-search-rag.md
parent_task_id: TASK-10.8
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Mount a Library-owned Search/RAG panel inside the existing Library three-pane shell instead of routing users to or embedding the legacy Search/RAG window.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Search/RAG mode mounts source scope query input run action evidence/results and retrieval inspector inside Library.
- [ ] #2 Switching into Search/RAG mode preserves the Library source browser detail and inspector shell selectors.
- [ ] #3 The legacy Search/RAG container is not embedded inside Library.
- [ ] #4 The existing standalone Search/RAG compatibility route action remains available and tested.
<!-- AC:END -->
