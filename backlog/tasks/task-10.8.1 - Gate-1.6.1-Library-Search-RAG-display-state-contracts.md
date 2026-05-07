---
id: TASK-10.8.1
title: 'Gate 1.6.1: Library Search/RAG display-state contracts'
status: Done
assignee: []
created_date: '2026-05-07 12:00'
labels:
  - product-maturity
  - phase-3-knowledge-study
  - gate-1-6-library-rag
updated_date: '2026-05-07 13:00'
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
1. Run the current Library contract-layout baseline and Gate 1 Library mode guardrail.
2. Add failing pure state tests for Library Search/RAG source scope query state result rows action states and panel statuses.
3. Implement minimal frozen dataclasses under tldw_chatbook/Library without Textual imports.
4. Verify the new unit tests and existing Library contract-layout tests.
5. Run diff hygiene before committing.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added pure Library Search/RAG display-state contracts in tldw_chatbook/Library/library_rag_state.py with scope query result action and panel state dataclasses. Covered source counts for notes media conversations Workspaces and Collections, empty query/source recovery copy, citation/snippet/provenance normalization, malformed result tolerance, and ready searching blocked empty panel status behavior. Added Tests/Library/test_library_rag_state.py and kept existing Library contract-layout guardrails green.
<!-- SECTION:NOTES:END -->
