---
id: TASK-10.8.1
title: 'Gate 1.6.1: Library Search/RAG display-state contracts'
status: Done
assignee: []
created_date: '2026-05-07 12:00'
updated_date: '2026-05-07 19:52'
labels:
  - product-maturity
  - phase-3-knowledge-study
  - gate-1-6-library-rag
dependencies:
  - TASK-10.7
documentation:
  - Docs/superpowers/specs/2026-05-06-screen-design-adaptation-audit-design.md
  - Docs/superpowers/plans/2026-05-07-gate-1-6-library-native-search-rag.md
parent_task_id: TASK-10.8
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create pure Library Search/RAG display-state contracts for source scope query state retrieval status evidence rows citations snippets actions and recovery copy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Pure state builders expose notes media conversations Workspaces and Collections source scope query mode retrieval status result rows citations snippets and action disabled reasons without Textual dependencies.
- [x] #2 Empty query no source missing index and unavailable dependency states produce visible recovery copy with owner and next action.
- [x] #3 Unit tests cover valid results malformed values empty states and blocked states.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing unit coverage for Library Search/RAG display-state contracts.
2. Verify the tests fail because the pure state module does not exist yet.
3. Implement the minimal non-Textual dataclasses/builders for source scope, query state, result rows, action state, and panel state.
4. Run the focused state tests plus the existing Library contract layout tests and diff hygiene.
5. Update task acceptance criteria and implementation notes after verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented pure Library Search/RAG display-state contracts in tldw_chatbook/Library/library_rag_state.py with source scope options for notes media conversations Workspaces and Collections, query blocking/recovery copy, normalized evidence rows with citations/snippets/provenance, and panel action readiness. Added Tests/Library/test_library_rag_state.py covering valid results malformed values empty states and blocked states. Kept the module independent of Textual imports.
<!-- SECTION:NOTES:END -->
