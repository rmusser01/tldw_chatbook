---
id: TASK-3304
title: >-
  Ingest structural fixes: disabled-state legibility, receipt into view, picker table, clipped install command
status: In Progress
assignee:
  - '@claude'
created_date: '2026-08-07 19:30'
labels:
  - library
  - ingest
  - ux
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Findings MI-07/08/15/17 of the 2026-08-07 Media Ingestion review. (1) All `enabled_when_values`-disabled option fields (Parakeet model folder, transcription model under `default`, web max pages/depth under `individual`) and the Parakeet install button render identically to enabled controls with no stated reason — violating the dense-form Inert-actions rule and sitting on the documented all-themes-below-3:1 disabled-contrast trap. (2) After Start, the outcome rows land below the fold on every submit and the `VerticalScroll` canvas has no fold indicator (task-1623 convention). (3) The file picker table has no column headers, raw unitless sizes, a `..` row showing "512" as an apparent size, and an unlabeled filename input. (4) The missing-dependency warning line clips the pip command at the canvas edge, and the only copy button lives in the guardrail modal.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Schema-disabled fields and the install button read as visibly inert with the reason at the control (e.g. "— needs Parakeet provider"), at ≥3:1 label contrast in a running terminal
- [ ] #2 After Start, the queue's outcome area is brought into view (or a pinned live status adjacent to Start shows run state until terminal); a fold indicator appears while canvas content overflows
- [ ] #3 Picker shows column headers, humanized sizes, no size on directory rows, and a labeled filename input
- [ ] #4 The full install command is readable (wrapped or truncated-with-copy affordance outside the modal)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-verify each on the worktree; RED tests where assertable (reason-annotation presence, post-start scroll target, picker header row).
2. Disabled: reason suffix from the schema's enabled_when metadata; app-tier disabled styling (Legible Disabled rule — app stylesheet, not DEFAULT_CSS).
3. Receipt: scroll-to-queue on submit + overflow fold row (display-managed, owned by the in-place updater — never conditionally composed).
4. Picker + warning-line copy affordance.
<!-- SECTION:PLAN:END -->
