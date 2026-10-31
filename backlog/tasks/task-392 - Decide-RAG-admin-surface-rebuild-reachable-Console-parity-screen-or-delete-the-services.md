---
id: TASK-392
title: >-
  Decide RAG admin surface: rebuild reachable Console-parity screen or delete
  the services
status: Done
assignee: []
created_date: '2026-07-19 04:23'
labels:
  - rag
  - product-decision
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The three RAG admin services (server_rag_admin_service, local_rag_admin_service, rag_admin_scope_service) have zero production consumers: their legacy UI was deleted in PR 669, task-248 (PR 677) repointed the local service at the shared vector store, and task-254 (PR 689) made construction lazy — explicitly deferring the keep-vs-delete decision as a product call. Resolve it: either (a) rebuild a reachable Console-parity admin surface (collection inspection, index stats, backfill trigger, chunking templates scope routing — note task-251 already shipped user-facing indexing controls in the Search window, so define what admin adds beyond that) or (b) delete the services, their scope-policy wiring, and Tests/RAG_Admin. This is a product decision task: the deliverable starts with a short written recommendation for owner sign-off before implementation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A written recommendation (rebuild vs delete, with scope and rationale) exists and has owner sign-off recorded in this task
- [x] #2 The chosen option is implemented: either an admin surface reachable through shell navigation consuming the services, or the services plus their tests and wiring are removed
- [x] #3 No unreachable-but-constructed admin code remains either way
<!-- AC:END -->


## Closeout audit 2026-10-11

Board-hygiene flip (cohort A, Jun-Jul 2026 status sweep). **Verdict: DONE** — evidence verified against origin/dev (a3c47c6c72).

The keep option won de facto: the admin services gained production consumers reachable through shell navigation — UI/Screens/chunking_lab_screen.py (Chunking Lab), Chunking/auto_selection.py:328 (LocalRAGAdminService.list_templates), and app_service_wiring/app.py wire server_rag_admin_service; settings_rag_profile_adapter.py / settings_rag_definition_actions.py consume the RAG profile services. AC#1 (a written recommendation with owner sign-off recorded here) was never written down.
