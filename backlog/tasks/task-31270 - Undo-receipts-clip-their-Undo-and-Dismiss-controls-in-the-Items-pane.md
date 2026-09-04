---
id: TASK-31270
title: Undo receipts clip their Undo and Dismiss controls in the Items pane
status: Done
assignee: []
created_date: '2026-09-04 13:54'
updated_date: '2026-09-04 15:09'
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
- [x] #1 Delete, bulk-delete and set-dismiss receipts paint Undo and Dismiss fully at the shell's Items-pane width (painted-text test at 38 cols)
- [x] #2 Receipts use the multi-row grammar: message row plus an action row, width 100%
- [x] #3 The message wraps or shortens without hiding either action
- [x] #4 Live-verified at 235x52 and 100x30
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: painted-text tests for both receipts at the real Items-pane width (235x52)
2. GREEN: both receipts become a Vertical (copy row + ds-toolbar action row) with unchanged button ids; width 100% rules in BUNDLED_CSS and the component sheet; bundle regenerated
3. Run render-fixes, side-by-side, multiselect, trash; live tmux 235x52 and 100x30
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The delete and set-dismiss receipts were single content-width Horizontals, so at the ~38-col Items pane they painted as '✓ deleted · 1 item · in Trash  Und' and '✓ dismissed · … Un' — Undo unfindable by label. Both are now a Vertical with a full-width copy Static (library-media-receipt-copy) and a ds-toolbar action row (library-media-receipt-actions) holding the same Undo/Dismiss buttons and ids, mirroring task-30043's multi-row toolbar grammar. CSS at both tiers (BUNDLED_CSS for the plain harnesses, _agentic_terminal.tcss for the app); bundle rebuilt and check_bundle_sync green. Painted-text tests pin both receipts at pane width. Live: 'Undo     Dismiss' on its own row under each receipt, Undo clickable by label, item and set restored; also at 100x30.
<!-- SECTION:NOTES:END -->
