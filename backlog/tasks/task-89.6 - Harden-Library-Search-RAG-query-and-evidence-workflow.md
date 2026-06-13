---
id: TASK-89.6
title: Harden Library Search RAG query and evidence workflow
status: Done
dependencies:
- TASK-89.1
labels:
- library
- search
- rag
- evidence
priority: high
parent_task_id: TASK-89
documentation:
- Docs/superpowers/specs/2026-05-23-citation-snippet-carry-through-epic-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make Library Search/RAG usable as a Library content-hub workflow: a user can enter a query, understand readiness, inspect evidence, and hand selected results into Console without losing citation/snippet context.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Library Search/RAG mode makes query entry, source scope, readiness, and run state visible before retrieval starts.
- [x] #2 Empty corpus, missing provider, missing embedding/index, and failed retrieval states explain what happened and offer a next action where possible.
- [x] #3 Retrieved results expose enough citation/snippet metadata for a user to decide what to use without opening unrelated screens.
- [x] #4 A selected result can be staged or handed off to Console with citation/snippet context preserved in the handoff payload.
- [x] #5 Keyboard-only operation covers query focus, run, result selection, detail inspection, and handoff.
- [x] #6 Evidence handoff behavior reuses the existing citation/snippet carry-through contract and does not introduce a competing citation payload shape.
- [x] #7 Focused regression coverage verifies blocked states, selected-result inspector content, and Console handoff payload shape.
- [x] #8 Actual CDP QA captures query entry, a blocked/recovery state, and at least one successful or fixture-backed evidence handoff screenshot before approval.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: This slice hardens existing Library Search/RAG UI state and reuses the existing citation/snippet handoff contract. It does not change storage/schema, provider/runtime ownership, sync policy, or a cross-module service boundary.

1. Inspect the current Library Search/RAG panel, state builder, service outcome handling, Console handoff payload builder, and existing Gate 1.6 tests.
2. Add failing regressions for blocked/recovery states, selected evidence inspector metadata, Console handoff payload shape, and keyboard-only operation.
3. Implement the smallest safe changes to make readiness, evidence inspection, and handoff metadata explicit without rebuilding Search/RAG architecture.
4. Run focused Library/Search-RAG tests and diff checks.
5. Capture actual CDP screenshots for query entry, blocked/recovery, and fixture-backed handoff before requesting approval.
<!-- SECTION:PLAN:END -->

## Non-Goals

<!-- SECTION:NON_GOALS:BEGIN -->
- Do not implement answer-level citation injection, citation persistence, Chatbook citation export, or RAG index architecture in this slice.
- Do not imply an answer is grounded unless valid evidence is selected and eligible for the active Console context.
<!-- SECTION:NON_GOALS:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added deterministic Library Search/RAG regressions covering query readiness, blocked recovery copy, selected evidence metadata, keyboard-only operation, and Console handoff payload shape.
- Hardened the Library-native Search/RAG panel so query controls, source scope, evidence rows, source/workspace/citation/eligibility badges, selected evidence inspection, and Console handoff controls remain visible without leaving Library.
- Reused the existing citation/snippet carry-through payload builder for Console handoff; no competing citation payload shape was introduced.
- Added explicit no-source recovery checklist copy and an Import/Export recovery action for empty Library corpora.
- Rebuilt generated Textual CSS from the source TCSS changes.
- Verification: `python -m pytest -q Tests/UI/test_product_maturity_gate16_library_search_rag.py --tb=short` passed with 16 tests and the existing `RequestsDependencyWarning`.
- Verification: `git diff --check` passed.
- Actual CDP/Textual-web approval evidence:
  - `Docs/superpowers/qa/product-maturity/screen-qa/library/task-89-6-library-rag-selected-evidence-polished-cdp-2026-06-09.png`
  - `Docs/superpowers/qa/product-maturity/screen-qa/library/task-89-6-library-rag-blocked-polished-cdp-2026-06-09.png`
  - User approved the rendered screenshots in-session on 2026-06-09.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Library Search/RAG now behaves like a Library content-hub workflow rather than a placeholder route: users can see query readiness, understand source-scope blockers, inspect source/citation/workspace eligibility, and stage selected evidence into Console with the existing citation/snippet handoff contract preserved. Focused regressions and approved actual CDP screenshots cover the changed workflow.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
