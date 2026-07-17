---
id: TASK-254
title: Stop constructing unreachable RAG_Admin services at every launch
status: To Do
assignee: []
created_date: '2026-07-12 14:12'
labels:
  - rag
  - cleanup
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
app.py:2545-2559 instantiates server_rag_admin_service, local_rag_admin_service and rag_admin_scope_service on every startup, but every UI consumer of these services (Embeddings_Management_Window, chunking-template widgets) is only mounted by the dead legacy SearchWindow stack and is unreachable. This adds startup cost and implies a live admin surface that does not exist. Either construct them lazily when a reachable surface actually needs them or expose a real reachable surface. Filed from the 2026-07-12 RAG module audit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 RAG admin services are no longer eagerly constructed at startup unless a reachable UI surface consumes them
- [ ] #2 Any surface that does consume them is reachable through shell navigation
- [ ] #3 App startup and test suite pass unchanged otherwise
<!-- AC:END -->
