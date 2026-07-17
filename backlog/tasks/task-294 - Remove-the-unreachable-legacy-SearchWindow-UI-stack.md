---
id: TASK-294
title: Remove the unreachable legacy SearchWindow UI stack
status: To Do
assignee: []
created_date: '2026-07-12 14:12'
labels:
  - rag
  - cleanup
  - ui
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UI/SearchWindow.py is imported by nothing after the master-shell redesign, yet it is the only mount point for SearchEmbeddingsWindow, Embeddings_Management_Window, the embeddings wizards and the chunking-template widgets - none of which are reachable from any registered route. UI/SearchRAGWindow.py.bak also lingers. Per the redesign rule legacy widgets are not reused unless fully redone Console-style; if manual embeddings management is still wanted it should be rebuilt as a new Console-parity screen in a separate task. Remove the dead stack so the codebase reflects the real UI surface. Filed from the 2026-07-12 RAG module audit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 UI/SearchWindow.py and screens/widgets reachable only through it (SearchEmbeddingsWindow, Embeddings_Management_Window, embeddings wizards, chunking-template widgets) are removed along with SearchRAGWindow.py.bak
- [ ] #2 No route registry or import references to the removed screens remain
- [ ] #3 Full test suite passes after removal
<!-- AC:END -->
