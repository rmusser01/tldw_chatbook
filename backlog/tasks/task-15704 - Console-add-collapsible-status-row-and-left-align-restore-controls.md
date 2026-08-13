---
id: TASK-15704
title: 'Console: add collapsible status row and left-align restore controls'
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-13 06:05'
updated_date: '2026-08-13 13:29'
labels:
  - console
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reduce Console visual noise by allowing the status-chip row above the composer to collapse while keeping restore controls immediately discoverable at the far left of both collapsed rows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 User can collapse the status-chip row to a one-line `Status hidden` presentation.
- [ ] #2 User can restore the status-chip row from a far-left `Status` control, and existing chip state is preserved.
- [ ] #3 The collapsed composer restore control appears at the far left while status copy and conditional `Stop` remain usable.
- [ ] #4 Status-row collapse state is screen-local and resets when the Console screen is recreated.
- [ ] #5 Keyboard focus order and narrow/wide geometry remain correct.
- [ ] #6 Relevant automated tests and live Textual verification pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing widget tests for mounted expanded/collapsed status-row presentations and preserved chip state.
2. Implement the minimal `ConsoleStatusChips.set_collapsed()` display toggle without recomposition.
3. Add failing screen tests, then wire screen-local state, button handlers, and inverse-control focus restoration in `ChatScreen`.
4. Update composer geometry expectations first, then move the existing `Expand ▴` child to the far left of the collapsed row.
5. Add and bundle minimal TCSS geometry rules; run focused regressions, layout detection, and isolated live Textual verification.
6. Complete acceptance criteria, implementation notes, and task closeout only after verification passes.

Detailed plan: `Docs/superpowers/plans/2026-08-13-console-row-collapse-controls.md`

Design: `Docs/superpowers/specs/2026-08-12-console-row-collapse-controls-design.md`

ADR required: no

ADR path: N/A

Reason: This is a screen-local UI behavior change following the existing composer ownership pattern; it changes no durable architecture boundary.
<!-- SECTION:PLAN:END -->
