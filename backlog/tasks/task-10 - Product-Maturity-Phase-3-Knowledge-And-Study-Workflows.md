---
id: TASK-10
title: 'Product Maturity Phase 3: Knowledge And Study Workflows'
status: In Progress
assignee: []
created_date: '2026-05-05 15:11'
updated_date: '2026-05-08 02:30'
labels:
  - product-maturity
  - phase-3-knowledge-study
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Mature ingest, organize, retrieve, study, and reuse workflows across Library, Workspaces, Library Collections, flashcards, quizzes, citations/snippets, and source evidence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 QA walkthrough verifies the running app is usable for this phase's target workflows.
- [ ] #2 Focused regression evidence exists for changed seams.
- [ ] #3 Repo-tracked QA evidence exists under Docs/superpowers/qa/product-maturity/.
- [ ] #4 P0/P1 findings are fixed or explicitly accepted according to the spec severity policy.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started Phase 3 with TASK-10.1 as the first PR-sized Knowledge and Study workflow gate. Parent remains open until later Phase 3 child gates verify import/export Search/RAG Workspaces flashcards quizzes and collections workflows end to end.

Continued Phase 3 with TASK-10.2. Library source context now carries into Study Dashboard Flashcards and Quizzes without changing deck/quiz service scope. Parent remains open for source-selected generation Workspaces Library Collections Import/Export Search/RAG citations/snippets and Console reuse gates.

Continued Phase 3 with TASK-10.3. The Library destination now exposes the approved Phase 3.0 layout shell with mode bar source browser source detail and inspector regions across compact default and large terminal sizes. Parent remains open for source-selected study generation Workspaces Library Collections and deeper Import/Export Search/RAG workflows.

Continued Phase 3 with TASK-10.4. Library-selected note and media source items now carry into Study Dashboard and can queue a server study-pack generation job with local-mode recovery. Parent remains open for completed study-pack polling/reuse conversation message-level selection Workspaces Library Collections citations/snippets and deeper Import/Export Search/RAG workflows.

Continued Phase 3 with TASK-10.5. Gate 1 now adapts Home Console and Library into the core product-loop screen model: Home Command Center regions, Console Agent Workbench regions around the existing chat surface, and actionable Library modes. Parent remains open for required Gate 1.5 Console internals decomposition, Gate 1.6 Library-native Search/RAG with citations/snippets, Library-owned Collections, and later Knowledge/Study workflow depth.

Continued Phase 3 with TASK-10.7. Queued server source study-pack jobs now observe bounded completion status and surface ready pack metadata as reusable Study dashboard state. Parent remains open for full server job history direct generated deck selection conversation message-level selection Workspaces Library Collections citations/snippets and deeper Import/Export Search/RAG workflows.

Closed Gate 1.5 with TASK-10.6. Console now uses native workbench internals for provider/model controls, staged context, transcript/session, composer, run inspector, approvals, RAG/source state, and Chatbook artifact actions instead of presenting a full embedded legacy Chat screen. Parent remains open for Gate 1.6 Library-native Search/RAG with citations/snippets, Workspaces, Library Collections, and deeper Import/Export/Search/RAG study workflows.

Planned Gate 1.6 with TASK-10.8. The Library-native Search/RAG gate is split into display-state contracts, a Library-owned Search/RAG panel, retrieval adapter and evidence normalization, Console handoff/Console-invoked RAG seams, and QA closeout. Parent remains open for execution of TASK-10.8 and later Workspaces, Library Collections, and deeper Import/Export/Search/RAG study workflows.

Continued Gate 1.6 with TASK-10.8.2. The Library destination now mounts a native Search/RAG panel for source scope, query controls, evidence/results, retrieval inspector, and Console handoff readiness without embedding the legacy Search/RAG route. Parent remains open for retrieval execution, evidence normalization, Console handoff/invocation, QA closeout, Workspaces, Library Collections, and deeper Import/Export/Search/RAG study workflows.

Closed Gate 1.6 with TASK-10.8. Library-native Search/RAG now verifies source scope, query controls, retrieval execution, evidence/results, citations/snippets, selected evidence review, Console staged evidence, Console-initiated Library RAG, and recoverable blocked states. Parent remains open for Workspaces, Library Collections, deeper Import/Export/Search/RAG study flows, and later server-parity depth.
<!-- SECTION:NOTES:END -->
