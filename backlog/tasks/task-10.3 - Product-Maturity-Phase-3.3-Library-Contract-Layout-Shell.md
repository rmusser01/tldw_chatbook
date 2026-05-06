---
id: TASK-10.3
title: 'Product Maturity Phase 3.3: Library Contract Layout Shell'
status: Done
assignee: []
created_date: '2026-05-06 06:22'
updated_date: '2026-05-06 06:27'
labels:
  - product-maturity
  - phase-3-knowledge-study
  - library-ux
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-05-06-destination-layout-ia-contracts-design.md
  - >-
    Docs/superpowers/plans/2026-05-06-product-maturity-phase-3-3-library-contract-layout.md
parent_task_id: TASK-10
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the Library destination visibly follow the approved Phase 3.0 layout contract so users can distinguish source browsing, Search/RAG, Import/Export, Workspaces, Study, Flashcards, Quizzes, source detail, and Console handoff without relying on legacy route knowledge.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Library exposes a contract-aligned mode bar with Sources Search/RAG Import/Export Workspaces Study Flashcards Quizzes.
- [x] #2 Library exposes source browser detail and inspector regions with stable selectors.
- [x] #3 Library keeps existing Notes Media Conversations Import/Export Search/RAG Study Flashcards Quizzes and Use in Console actions reachable and labelled.
- [x] #4 Focused mounted regression tests verify compact default and large Library layout contract essentials.
- [x] #5 Repo-tracked QA evidence and roadmap/backlog updates record the Phase 3.3 verification and any residual risk.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add mounted Library contract layout regressions for the Phase 3.0 mode bar source browser detail inspector and existing action selectors.
2. Reshape LibraryScreen composition only preserving current service snapshot and action handler behavior.
3. Verify compact default and large Library layout essentials plus existing Phase 3.1 Phase 3.2 and destination-shell regressions.
4. Record Phase 3.3 QA evidence tracker updates and Backlog task hygiene.
5. Run focused verification and git diff hygiene before PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the Phase 3.3 Library contract layout shell. LibraryScreen now exposes the approved mode bar source browser source detail and source inspector regions while preserving existing Notes Media Conversations Import/Export Search/RAG Study Flashcards Quizzes and Use in Console actions. Focused mounted regressions verify compact default and large terminal layout essentials and caught an intermediate wide-mode-bar out-of-bounds regression; final layout keeps action buttons in vertical regions. Added Phase 3.3 QA evidence roadmap README and Backlog tracking.
<!-- SECTION:NOTES:END -->
