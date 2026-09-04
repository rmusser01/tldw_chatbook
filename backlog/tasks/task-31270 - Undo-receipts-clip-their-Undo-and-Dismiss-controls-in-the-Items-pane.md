---
id: TASK-31270
title: Undo receipts clip their Undo and Dismiss controls in the Items pane
status: To Do
assignee: []
created_date: '2026-09-04 13:54'
labels:
  - library
  - media-ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 P1: the delete and dismiss receipts render as a single content-width row, so in the shell's ~38-col Items pane they read `✓ deleted · 1 item · in Trash  Und` and `✓ dismissed · 2 selected items  Un` (B cap_99, cap_83). Undo is unfindable by label and the trailing Dismiss button is off-pane; recovery only worked by clicking the visible `U` cell. The toolbars were converted to the multi-row grammar in #2350; the receipts were not. An unreadable Undo is a hidden recovery state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Delete, bulk-delete and set-dismiss receipts paint Undo and Dismiss fully at the shell's Items-pane width (painted-text test at 38 cols)
- [ ] #2 Receipts use the multi-row grammar: message row plus an action row, width 100%
- [ ] #3 The message wraps or shortens without hiding either action
- [ ] #4 Live-verified at 235x52 and 100x30
<!-- AC:END -->
