---
id: TASK-142
title: Correct Console Workbench visual density and visible controls
status: Done
assignee:
  - '@codex'
created_date: '2026-06-29 19:38'
updated_date: '2026-06-29 19:55'
labels:
  - ui
  - textual
  - workbench
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Correct the first Console Workbench implementation so the redesigned frame visibly changes the working UI instead of adding passive space. The Console should show compact, purposeful controls and action-first inspector/composer regions while preserving the existing Workbench framing and responsiveness gates.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Console exposes provider, model, persona, RAG, source, tools, approvals, and core actions as visible compact controls without relying on the command palette.
- [x] #2 Passive empty space in the Console frame is reduced or converted into state, action, or content-bearing regions.
- [x] #3 Inspector presents run recovery, approvals, tools, evidence, and artifact actions in an action-first dense layout.
- [x] #4 Composer remains bottom-pinned and visibly supports attach, send, stop, save, and setup recovery states.
- [x] #5 Visual proof includes refreshed normal, compact, focus, and command-palette screenshots plus before-after QA notes.
- [x] #6 Targeted Console Workbench contract, visual snapshot, parity, responsiveness, and CSS checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no

ADR path: `backlog/decisions/011-chatbook-workbench-ui-system.md`

Reason: this is a corrective implementation of the accepted Workbench UI System decision, especially its visible workflow controls and command-palette discoverability constraints.

Plan: `Docs/superpowers/plans/2026-06-29-console-workbench-density-correction-plan.md`

1. Add failing tests that reject a single passive summary line and require visible compact provider/model/persona/RAG/source/tools/approvals controls.
2. Add failing tests that require action-first inspector content and visible composer action/recovery affordances.
3. Replace the Console control-bar summary with state chips and compact action affordances while keeping compatibility selectors mounted.
4. Rework inspector ordering and CSS density so the right rail shows actionable run/source/tool/artifact state before secondary headings.
5. Refresh visual artifacts and QA notes with before-after evidence.
6. Run targeted Workbench contract, visual snapshot, parity, responsiveness, CSS, and diff hygiene checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a corrective Console Workbench density pass after visual review found the first slice too close to the legacy UI. Console now keeps a one-row header, renders provider/model/persona/RAG/source/tool/approval state as visible compact chips, adds visible control-strip actions for Settings, Attach, Library RAG, and Help, prioritizes run/source/tool/approval inspector groups before secondary session detail, and shows composer setup recovery beside the bottom-pinned draft actions.

The pass preserves legacy control-bar selectors and the hidden CompactModelBar sync seam for existing Console tests while making useful controls visible. Visual artifacts were regenerated and now show the dense control strip and composer recovery state. Verification covered TDD red/green tests, targeted Workbench/Console UI suites, selected legacy decomposition seams, CSS build, route-switch soak, visual artifact checks, and diff hygiene.
<!-- SECTION:NOTES:END -->
