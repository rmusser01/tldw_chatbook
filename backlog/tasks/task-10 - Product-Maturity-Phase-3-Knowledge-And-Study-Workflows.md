---
id: TASK-10
title: 'Product Maturity Phase 3: Knowledge And Study Workflows'
status: In Progress
assignee: []
created_date: '2026-05-05 15:11'
updated_date: '2026-05-06 06:27'
labels:
  - product-maturity
  - phase-3-knowledge-study
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Mature ingest, organize, retrieve, study, and reuse workflows across Library, Workspaces, flashcards, quizzes, and collections.
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

Continued Phase 3 with TASK-10.2. Library source context now carries into Study Dashboard Flashcards and Quizzes without changing deck/quiz service scope. Parent remains open for source-selected generation Workspaces Collections Import/Export Search/RAG and Console reuse gates.

Continued Phase 3 with TASK-10.3. The Library destination now exposes the approved Phase 3.0 layout shell with mode bar source browser source detail and inspector regions across compact default and large terminal sizes. Parent remains open for source-selected study generation Workspaces Collections and deeper Import/Export Search/RAG workflows.

Continued Phase 3 with TASK-10.4. Library-selected note and media source items now carry into Study Dashboard and can queue a server study-pack generation job with local-mode recovery. Parent remains open for completed study-pack polling/reuse conversation message-level selection Workspaces Collections and deeper Import/Export Search/RAG workflows.
<!-- SECTION:NOTES:END -->
