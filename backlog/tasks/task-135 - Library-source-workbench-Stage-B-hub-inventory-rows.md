---
id: TASK-135
title: Library source workbench Stage B hub inventory rows
status: Done
labels:
- library
- ux
- source-workbench
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace the Library hub prose with structured source inventory rows so users can quickly understand source readiness, ownership, primary actions, next-best action, and blocked Console handoff state without adding new service calls.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Library hub renders structured inventory rows for Notes, Media, Conversations, Collections, Search/RAG, Import/Export, and Study using existing local state only.
- [x] #2 Each inventory row exposes count or readiness, owner responsibility, and primary action copy without changing route IDs or service boundaries.
- [x] #3 The hub chooses one visible next-best action for the empty/default state and keeps Console handoff visibly disabled until eligible source context exists.
- [x] #4 Mounted regressions cover inventory row labels, counts/readiness, owner copy, next-best action, disabled handoff, and no new server dependency assumptions.
- [x] #5 QA evidence includes an actual CDP/Textual-web screenshot of the updated Library hub and a note that Stage B does not introduce tldw_server runtime calls.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:IMPLEMENTATION_PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: Stage B changes Library hub presentation, copy, tests, and QA evidence only; it does not change storage/schema, sync policy, provider/runtime boundaries, service contracts, security, or dependencies.

1. Add a failing mounted regression for Stage B inventory rows covering source labels, counts/readiness, owners, primary actions, next-best action, disabled Console handoff, and no server-runtime assumptions.
2. Replace the hub prose/table hybrid with a structured inventory section using existing Library snapshot state only.
3. Keep route IDs, owner buttons, service boundaries, and Stage A shell behavior unchanged.
4. Run focused Library regressions, roadmap/layout contract tests, and `git diff --check`.
5. Capture actual Textual-web/CDP screenshots and document QA evidence before PR handoff.
<!-- SECTION:IMPLEMENTATION_PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented Stage B as structured Library hub inventory rows using existing local source snapshot state. Added compact rows for Notes, Media, Conversations, Collections, Search/RAG, Import/Export, and Study with readiness, owner, primary action, and Console handoff state. Preserved existing route IDs and owner buttons. Added mounted regression coverage for the new row contract and updated older hub assertions away from the replaced framed table. QA evidence was captured from actual Textual-web/CDP output and approved by the user. No `tldw_server` runtime calls or new service dependencies were introduced.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Library Source Workbench Stage B verified: the Content Hub now presents a complete source inventory, blocked Console handoff remains explicit, and visual evidence is recorded under `Docs/superpowers/qa/library-source-workbench-stage-b/`.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
