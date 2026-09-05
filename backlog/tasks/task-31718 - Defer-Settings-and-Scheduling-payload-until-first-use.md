---
id: TASK-31718
title: Defer Settings and Scheduling payload until first use
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 17:43'
updated_date: '2026-09-05 18:06'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Repay screen pre-import payload growth by deferring Settings RAG and Tool Pack services plus Scheduling detail and modal implementations until their actual features are used.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Settings route pre-import does not eagerly load the RAG profile engine or Tool Pack service/review implementation closure.
- [x] #2 Scheduling route pre-import defers detail widgets and create/edit/audit forms until their actual first use, preserving event identity and runtime behavior.
- [x] #3 Focused import-closure regressions and affected Settings/Scheduling behavior tests pass without ratchet increases or exemptions.
- [x] #4 The complete pre-import census records measured reductions and any remaining independently owned Library cost.
- [x] #5 The compact Scheduling geometry harness loads the canonical app stylesheet set, preserving its original one-row and visibility assertions.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add fresh-interpreter regression coverage for the measured Settings and Scheduling import edges and demonstrate failures before implementation.
2. Defer Settings RAG profile adapter calls and Tool Pack service/review imports; retain canonical ToolProfilesPanel event identity and quote moved DTO annotations where required.
3. Move Scheduling detail/widget/form imports into actual first-use paths, retaining canonical helper behavior and compatibility call seams.
4. Correct the pre-existing compact Scheduling harness to load APP_STYLESHEETS: paired original/current workbench runs fail without the owning sheet, and an in-memory canonical-style run passes unchanged assertions.
5. Run focused import and all 482 affected behavioral tests, complete pre-import census, scoped static checks, and review. Coordinate Library-owned route reduction separately; do not edit Library or Console modules.
ADR required: no new ADR
ADR path: backlog/decisions/097-boot-budget-ratchets.md
Reason: accepted first-use deferral policy and faithful test-harness preconditions; public runtime boundaries and all existing limits remain unchanged.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Deferred Settings RAG/Tool Pack implementations and Scheduling detail/forms until first use under ADR-097, preserving explicit patchable Settings wrappers, canonical nested-event class identity, and exact cached Scheduling re-exports. Added isolated RED-to-GREEN closure checks, including repeat identity and unknown-export behavior. Corrected only the compact Scheduling test harness to load canonical APP_STYLESHEETS; identical original/current failures and an in-memory canonical-style pass established the pre-existing missing-sheet cause, and original geometry assertions remain unchanged. Verification: all 482 affected behavior cases passed in 190.20s; isolated closure plus unchanged census tests 3 passed in 8.80s. Baseline census539 modules/416278 LOC/146805 Library LOC; this task alone499/387596/146804; coordinated final489/371218/130426, under unchanged500/380000/145000 caps. Ruff lint on all owned Python files, formatting of new and changed ranges, git diff --check, normalized method-body AST comparison, and parent independent review passed. No ratchets, exemptions, product CSS, Console, or Library files changed. Reports: /private/tmp/tldw-31660-settings-scheduling-final.xml and /private/tmp/tldw-31660-final-census.xml. No new ADR required: direct implementation of backlog/decisions/097-boot-budget-ratchets.md. No new generalized lesson beyond existing canonical-stylesheet guidance.
<!-- SECTION:NOTES:END -->
