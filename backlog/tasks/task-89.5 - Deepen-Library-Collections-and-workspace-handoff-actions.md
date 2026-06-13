---
id: TASK-89.5
title: Deepen Library Collections and workspace handoff actions
status: Done
dependencies:
- TASK-89.1
labels:
- library
- collections
- workspaces
- handoffs
priority: medium
parent_task_id: TASK-89
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make Library Collections behave as actionable content sets inside the Library content-hub model, with clear workspace eligibility and downstream handoff affordances. Collections should help users understand what a grouped content set can do without implying that workspace switching hides global Library content.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A user can select a Collection from the Library Collections mode and see contextual details that explain its source membership and available actions.
- [x] #2 Collection-scoped actions clearly distinguish available local actions from deferred or WIP server/sync-backed behavior.
- [x] #3 Workspace rules are visible: global Library items remain discoverable, but staging/manipulation from Console is constrained by the current workspace context.
- [x] #4 Blocked or unavailable handoff actions show recoverable guidance instead of silent no-ops.
- [x] #5 Collection behavior follows the content-hub contract from TASK-89.1 and does not redefine workspace visibility or handoff semantics locally.
- [x] #6 Focused regression coverage verifies collection selection, inspector copy, action availability, and workspace gating.
- [x] #7 Actual CDP QA captures the rendered Collections state and receives screenshot approval before the task is considered complete.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: This slice only changes destination-native Library Collections UI copy, action affordances, and regression coverage. It does not introduce storage/schema changes, sync policy, service contracts, or new workspace ownership semantics.

1. Inspect the existing Library Collections screen, state models, services, and current content-hub tests.
2. Add focused failing mounted regressions for Collections selection, inspector copy, action availability, blocked/WIP guidance, and workspace gating language.
3. Implement the smallest UI/state changes needed to expose source membership, local-vs-deferred actions, and workspace visibility/staging boundaries without adding sync behavior.
4. Rerun focused Library tests plus existing Library/workspace contract coverage.
5. Capture a real Textual-web/CDP screenshot of the rendered Collections state and update QA notes.
6. After screenshot approval, check acceptance criteria, add implementation notes, and mark the task Done.
<!-- SECTION:PLAN:END -->

## Non-Goals

<!-- SECTION:NON_GOALS:BEGIN -->
- Do not implement server sync, shared collections, or collection membership migration in this slice.
- Do not make active workspace switching hide Collections or Library items from global browsing.
<!-- SECTION:NON_GOALS:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added mounted regressions for selected and empty Collections states covering membership copy, workspace gating, local/deferred action language, blocked handoff guidance, and disabled later-stage actions.
- Updated the Library Collections mode to refresh its status row after async snapshot load, expose selected Collection membership and workspace-boundary guidance, and keep collection-scoped Study/Flashcards/Quizzes/Console paths visibly blocked instead of silent.
- Updated the Collections detail panel and inspector to explain that Library items remain globally discoverable while active-workspace context controls staging/manipulation.
- Captured and received approval for the rendered CDP screenshot at `Docs/superpowers/qa/product-maturity/screen-qa/library/collections-workflow-cdp-2026-06-09.png`.
- ADR check: no ADR required because this slice only changes UI copy, action affordances, QA evidence, and regressions; no storage, schema, sync, service contract, or ownership boundary changed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Deepened Library Collections as a destination-native content-hub workflow with explicit membership details, workspace boundaries, available-vs-deferred actions, blocked handoff recovery, regression coverage, and approved rendered screenshot evidence.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
