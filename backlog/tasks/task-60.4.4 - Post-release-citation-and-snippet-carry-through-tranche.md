---
id: TASK-60.4.4
title: Post-release citation and snippet carry-through tranche
status: In Progress
labels:
- post-release
- rag
- citations
- ux
priority: medium
parent_task_id: TASK-60.4
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Carry retrieved snippets and citations from Library/Search/RAG into Console answers, Artifacts, exported Chatbooks, and downstream saved work only after source authority is visible and testable end-to-end.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Citation and snippet carry-through scope references TASK-60.3 actual-use audit evidence.
- [ ] #2 Retrieved evidence keeps source identity, snippet text, and authority labels through Console responses and saved artifacts.
- [ ] #3 Exported Chatbooks preserve citations/snippets in a user-readable and machine-checkable form.
- [ ] #4 QA verifies source-to-answer-to-artifact carry-through with actual app use before completion.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Open an epic tracking PR from `codex/citations-snippets-epic` to `dev` with the approved stacked-PR design.
2. Land the evidence/citation contract in a sub-PR targeting the epic branch.
3. Land Library/Search-RAG evidence bundle generation in a sub-PR targeting the epic branch.
4. Land Console evidence staging and blocked-state feedback in a sub-PR targeting the epic branch.
5. Land answer-level citation injection, response parsing, and validation in a sub-PR targeting the epic branch.
6. Land message persistence for evidence bundles and citation refs in a sub-PR targeting the epic branch.
7. Land Chatbook artifact/export evidence preservation in a sub-PR targeting the epic branch.
8. Close the epic with actual CDP/textual-web QA evidence, user-approved screenshots, focused automated tests, roadmap updates, and completed Definition of Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Epic setup started. Added `Docs/superpowers/specs/2026-05-23-citation-snippet-carry-through-epic-design.md` to define the stacked PR model, evidence/citation contracts, answer-level citation injection scope, persistence/export requirements, and QA gates. The implementation remains split across sub-PRs so each seam can be tested and reviewed independently.

Library/Search-RAG evidence bundle slice added. `build_library_rag_console_live_work_payload()` now attaches a serialized `EvidenceBundle` that preserves query, source identity, snippets, citation labels, local/server authority, workspace visibility, and active-context eligibility. Regression coverage verifies local and server Console handoff payloads, cross-workspace blocked evidence, and provenance-only source identity fallback before the later Console staging and answer-citation slices.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
