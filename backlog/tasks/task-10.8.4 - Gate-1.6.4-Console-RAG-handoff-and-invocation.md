---
id: TASK-10.8.4
title: 'Gate 1.6.4: Console RAG handoff and invocation'
status: To Do
assignee: []
created_date: '2026-05-07 12:00'
labels:
  - product-maturity
  - phase-3-knowledge-study
  - gate-1-6-library-rag
dependencies:
  - TASK-10.8.3
documentation:
  - Docs/superpowers/plans/2026-05-07-gate-1-6-library-native-search-rag.md
parent_task_id: TASK-10.8
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Preserve Library Search/RAG evidence when users continue into Console and expose a Console-owned way to invoke RAG against Library sources with visible retrieval state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Library Search/RAG results stage into Console with query source authority citations snippets score and recovery copy preserved.
- [ ] #2 Console can invoke Library RAG or show a recoverable blocked state when retrieval is unavailable.
- [ ] #3 Existing Search/RAG handoff and Console live-work tests remain green.
- [ ] #4 Library handoff actions remain disabled with specific reason copy until a query and usable evidence are available.
<!-- AC:END -->
