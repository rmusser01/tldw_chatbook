---
id: TASK-201
title: Wire or remove the dead Start Indexing controls in SearchRAGWindow
status: To Do
assignee: []
created_date: '2026-07-12 14:11'
labels:
  - rag
  - ux
dependencies:
  - TASK-197
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The standalone Search window (UI/Views/RAGSearch/search_rag_window.py) renders Index New Content controls whose start-indexing button (line 474) has no event handler anywhere in the codebase, along with index-stats controls that never update. The UI advertises indexing that cannot happen. Once a real bulk-index path exists (task-197) these controls should trigger it and reflect real index stats; otherwise they must be removed. Filed from the 2026-07-12 RAG module audit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Search window contains no controls that do nothing when activated
- [ ] #2 If indexing controls remain they trigger real indexing and display actual index statistics
<!-- AC:END -->
