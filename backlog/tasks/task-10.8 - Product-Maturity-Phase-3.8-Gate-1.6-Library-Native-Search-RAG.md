---
id: TASK-10.8
title: 'Product Maturity Phase 3.8: Gate 1.6 Library Native Search/RAG'
status: To Do
assignee: []
created_date: '2026-05-07 12:00'
labels:
  - product-maturity
  - phase-3-knowledge-study
  - gate-1-6-library-rag
dependencies:
  - TASK-10.6
  - TASK-10.7
parent_task_id: TASK-10
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Plan and execute the required Gate 1.6 Library-native Search/RAG workflow so users can retrieve cited source evidence from Library and continue into Console without relying on the legacy Search/RAG route.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Library Search/RAG mode exposes source scope query input retrieval status evidence/results citations/snippets and setup/error recovery inside the Library shell.
- [ ] #2 Library Search/RAG results can stage evidence into Console with source authority citations/snippets and recovery copy preserved.
- [ ] #3 Console can invoke RAG against Library sources with visible retrieval state or explicit recovery.
- [ ] #4 Focused automated and QA walkthrough evidence prove the workflow is usable rather than only selectable.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Gate 1.6 is split into child tasks so retrieval state, Library UI, adapter behavior, Console handoff, and QA closeout can ship in reviewable slices:

1. TASK-10.8.1: Library Search/RAG display-state contracts.
2. TASK-10.8.2: Library-native Search/RAG panel.
3. TASK-10.8.3: Retrieval adapter and evidence results.
4. TASK-10.8.4: Console handoff and Console-initiated RAG.
5. TASK-10.8.5: QA closeout and tracking.

Primary implementation plan: `Docs/superpowers/plans/2026-05-07-gate-1-6-library-native-search-rag.md`.
<!-- SECTION:PLAN:END -->
