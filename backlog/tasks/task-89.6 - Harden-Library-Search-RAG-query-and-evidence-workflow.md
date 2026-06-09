---
id: TASK-89.6
title: Harden Library Search RAG query and evidence workflow
status: To Do
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
- [ ] #1 The Library Search/RAG mode makes query entry, source scope, readiness, and run state visible before retrieval starts.
- [ ] #2 Empty corpus, missing provider, missing embedding/index, and failed retrieval states explain what happened and offer a next action where possible.
- [ ] #3 Retrieved results expose enough citation/snippet metadata for a user to decide what to use without opening unrelated screens.
- [ ] #4 A selected result can be staged or handed off to Console with citation/snippet context preserved in the handoff payload.
- [ ] #5 Keyboard-only operation covers query focus, run, result selection, detail inspection, and handoff.
- [ ] #6 Evidence handoff behavior reuses the existing citation/snippet carry-through contract and does not introduce a competing citation payload shape.
- [ ] #7 Focused regression coverage verifies blocked states, selected-result inspector content, and Console handoff payload shape.
- [ ] #8 Actual CDP QA captures query entry, a blocked/recovery state, and at least one successful or fixture-backed evidence handoff screenshot before approval.
<!-- AC:END -->

## Non-Goals

<!-- SECTION:NON_GOALS:BEGIN -->
- Do not implement answer-level citation injection, citation persistence, Chatbook citation export, or RAG index architecture in this slice.
- Do not imply an answer is grounded unless valid evidence is selected and eligible for the active Console context.
<!-- SECTION:NON_GOALS:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
