---
id: TASK-10.6
title: 'Product Maturity Phase 3.6: Gate 1.5 Console Internals Decomposition'
status: In Progress
assignee: []
created_date: '2026-05-07 03:36'
updated_date: '2026-05-07 04:25'
labels:
  - product-maturity
  - phase-3-knowledge-study
  - gate-1-5-console
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-05-06-screen-design-adaptation-audit-design.md
  - Docs/superpowers/specs/2026-05-06-destination-layout-ia-contracts-design.md
  - >-
    Docs/superpowers/plans/2026-05-07-gate-1-5-console-internals-decomposition.md
parent_task_id: TASK-10
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Plan and execute the required Gate 1.5 Console internals decomposition so Console becomes a coherent agentic workbench instead of a shell wrapped around legacy ChatWindowEnhanced internals.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Console-native components replace or isolate legacy ChatWindowEnhanced internals for provider/model controls, staged context, transcript, composer, run inspector, approvals, tools, RAG controls, and artifact actions.
- [ ] #2 Existing chat behavior remains compatible or has documented replacement coverage for basic chat, tabs/session state, provider/model selection, streaming fallback, handoffs, RAG-related controls, tool-call visibility, and persona/character attachment paths.
- [ ] #3 Mounted UI regressions verify Console internals fit the approved agentic terminal design system without presenting an out-of-place embedded legacy chat screen.
- [ ] #4 QA walkthrough verifies the Console is usable for repeated core-loop work not merely renderable or clickable.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Gate 1.5 is intentionally split into child tasks so the Console rewrite can proceed in reviewable slices:

1. Complete `TASK-10.6.1` by adding pure Console display-state contracts and red mounted guardrails for legacy embedded chrome.
2. Complete `TASK-10.6.2` by moving provider/model/persona/RAG/source controls and staged context into Console-owned widgets.
3. Complete `TASK-10.6.3` by replacing the full embedded `ChatWindowEnhanced` surface with native transcript/session and composer widgets that reuse existing chat services.
4. Complete `TASK-10.6.4` by moving run inspector approvals tools RAG state and Chatbook artifact actions into native Console inspector/action seams.
5. Complete `TASK-10.6.5` by replaying Console usability and compatibility QA, recording evidence, updating the roadmap, and closing this parent gate only when the app is actually usable.

Detailed step-by-step implementation instructions live in `Docs/superpowers/plans/2026-05-07-gate-1-5-console-internals-decomposition.md`.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Completed TASK-10.6.1. Console display-state contracts now exist as pure non-Textual dataclasses with focused tests, and strict xfailed mounted guardrails document the remaining embedded legacy ChatWindowEnhanced chrome for later Gate 1.5 slices.
<!-- SECTION:NOTES:END -->
