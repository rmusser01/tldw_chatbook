---
id: TASK-10.8.4
title: 'Gate 1.6.4: Console RAG handoff and invocation'
status: Done
assignee: []
created_date: '2026-05-07 12:00'
updated_date: '2026-05-08 02:27'
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
- [x] #1 Library Search/RAG results stage into Console with query source authority citations snippets score and recovery copy preserved.
- [x] #2 Console can invoke Library RAG or show a recoverable blocked state when retrieval is unavailable.
- [x] #3 Existing Search/RAG handoff and Console live-work tests remain green.
- [x] #4 Library handoff actions remain disabled with specific reason copy until a query and usable evidence are available.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Run the focused baseline for Library Search/RAG, Console internals, Search handoffs, and Console live-work handoffs.
2. Add red Library mounted UI tests proving a selected Library RAG result stages into Console live work with query, authority, citations, snippets, score, runtime backend, and review recovery copy.
3. Add red Console seam tests proving Console can request Library RAG retrieval or surface a persistent recovery state when the retrieval service is unavailable.
4. Implement minimal Library result selection, live-work launch payload construction, and Console Library RAG invocation state without replacing ChatWindowEnhanced internals.
5. Run focused verification, update plan/backlog tracking, and commit the slice.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added a shared Library Search/RAG to Console live-work payload builder that preserves query, source IDs, chunk ID, snippets, citations, score, runtime backend, and source authority.
- Added mounted Library UI selection and Console launch wiring so evidence remains disabled until a usable result is selected and then stages into Console with review recovery copy.
- Added a Console-owned Library RAG query seam that can retrieve against the visible local Library scope or stage a persistent blocked recovery state when the retrieval service is unavailable.
- Verified with the Task 4 focused suite: `90 passed, 1 warning`.
<!-- SECTION:NOTES:END -->
