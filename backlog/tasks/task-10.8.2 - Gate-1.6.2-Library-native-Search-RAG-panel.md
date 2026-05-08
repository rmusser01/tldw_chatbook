---
id: TASK-10.8.2
title: 'Gate 1.6.2: Library native Search/RAG panel'
status: Done
assignee: []
created_date: '2026-05-07 12:00'
updated_date: '2026-05-08 00:43'
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
- [x] #1 Search/RAG mode mounts source scope query input run action evidence/results and retrieval inspector inside Library.
- [x] #2 Switching into Search/RAG mode preserves the Library source browser detail and inspector shell selectors.
- [x] #3 The legacy Search/RAG container is not embedded inside Library.
- [x] #4 The existing standalone Search/RAG compatibility route action remains available and tested.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add mounted red tests proving Library Search/RAG mode stays inside Library and exposes native panel selectors while preserving the compatibility route action.
2. Implement a minimal Textual LibrarySearchRagPanel widget backed by LibraryRagPanelState.
3. Wire LibraryScreen search mode to render the native panel inside the existing Library detail and inspector shell without embedding SearchRAGWindow.
4. Run focused UI/layout verification plus diff hygiene.
5. Check acceptance criteria and add implementation notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Mounted a minimal Library-native Search/RAG panel across the existing Library detail and inspector panes. The detail pane renders source scope, query input, run action, and evidence/results; the inspector pane renders retrieval status and Console handoff readiness from LibraryRagPanelState. The standalone Search/RAG route button remains as a compatibility fallback, and mode switching mounts or removes the native Search/RAG regions without recomposing the Library source browser, detail, or inspector panes.
<!-- SECTION:NOTES:END -->
