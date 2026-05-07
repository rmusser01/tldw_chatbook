---
id: TASK-10.8.3
title: 'Gate 1.6.3: Retrieval adapter and evidence results'
status: To Do
assignee: []
created_date: '2026-05-07 12:00'
labels:
  - product-maturity
  - phase-3-knowledge-study
  - gate-1-6-library-rag
dependencies:
  - TASK-10.8.2
documentation:
  - Docs/superpowers/plans/2026-05-07-gate-1-6-library-native-search-rag.md
parent_task_id: TASK-10.8
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a retrieval adapter seam for Library Search/RAG that normalizes local or server results into cited evidence rows and persistent recovery states.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Adapter normalizes query results with source id chunk id snippet citations score runtime backend and provenance metadata.
- [ ] #2 Missing dependency unavailable service policy-denied empty result and failed retrieval states render persistent recovery instead of transient alerts.
- [ ] #3 Retrieval results are applied to Textual UI on the message thread and covered by focused tests.
<!-- AC:END -->
