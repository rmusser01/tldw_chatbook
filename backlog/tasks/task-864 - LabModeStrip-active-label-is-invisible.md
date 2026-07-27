---
id: TASK-864
title: >-
  LabModeStrip's active label is invisible — a one-row strip inheriting a
  three-row border
status: To Do
assignee: []
created_date: '2026-07-26 23:20'
labels:
  - ui
  - bug
  - css
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The active mode's label does not render in `LabModeStrip`. The user cannot see which mode is selected.

The mechanism, confirmed while fixing the identical bug in the Watchlists tab strip: the strip pins itself to `height: 1`, while the global `.is-active` rule in `css/components/_agentic_terminal.tcss` applies `border: round $ds-action-focus`. A bordered button is three rows tall, so inside a one-row strip only its top border survives — the row that would have held the text is clipped away.

This is the second occurrence of the bug class. The Watchlists fix (Phase C, task 6) added a strip-scoped rule with enough specificity to beat the global single-class selector — `border: none` plus `text-style: bold underline` — modelled on the MCP mode strip, which does not have the bug. Do not edit the shared `.is-active` rule; other screens depend on the border it draws.

Worth checking whether any other one-row strip inherits the same rule.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 The active mode's label is legible in `LabModeStrip`
- [ ] #2 The active mode is visually distinguishable from the inactive ones
- [ ] #3 The strip stays one row tall
- [ ] #4 A test fails when the active label is not visible, asserting against the compositor rather than `render_line()`
- [ ] #5 `css/components/_agentic_terminal.tcss`'s shared `.is-active` rule is unchanged
- [ ] #6 Every other one-row strip carrying `.is-active` has been checked for the same defect, and any found are listed here or fixed
<!-- AC:END -->
