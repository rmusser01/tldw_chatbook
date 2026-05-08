---
id: TASK-10.8.3
title: 'Gate 1.6.3: Retrieval adapter and evidence results'
status: Done
assignee: []
created_date: '2026-05-07 12:00'
updated_date: '2026-05-08 01:56'
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
- [x] #1 Adapter normalizes query results with source id chunk id snippet citations score runtime backend and provenance metadata.
- [x] #2 Missing dependency unavailable service policy-denied empty result and failed retrieval states render persistent recovery instead of transient alerts.
- [x] #3 Retrieval results are applied to Textual UI on the message thread and covered by focused tests.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Run the focused Library Search/RAG baseline from fresh origin/dev.
2. Add red service tests for result normalization, unavailable service recovery, policy-denied recovery, failed retrieval recovery, and empty result handling.
3. Implement a minimal LibraryRagSearchService protocol, adapter result model, and adapter runner that uses app_instance.library_rag_search_service when present and returns persistent recovery state otherwise.
4. Add red mounted UI tests for running a Library Search/RAG query through a fake service and rendering evidence/results, searching state, and recovery states.
5. Wire LibraryScreen run action to execute retrieval via a worker and apply results on the Textual message thread.
6. Run focused verification, update plan/backlog/tracker state, and commit the slice.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a LibraryRagSearchService seam with request/outcome contracts, result normalization into LibraryRagResultRow, and stable DestinationRecoveryState output for unavailable service, policy denial, empty result, and failed retrieval states. LibraryScreen now runs Library Search/RAG through a worker, applies outcomes back to the native panel, and renders evidence titles, snippets, scores, citations, retrieval status, and persistent recovery selectors. Focused service and mounted UI regressions cover the adapter behavior and visible Library execution path.
<!-- SECTION:NOTES:END -->
