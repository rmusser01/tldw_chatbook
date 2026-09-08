---
id: TASK-31980
title: 'Library media: the permanent-delete button carries no danger affordance'
status: Done
assignee: []
created_date: '2026-09-07 22:48'
updated_date: '2026-09-08 04:35'
labels:
  - library
  - media
  - ux
  - css
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #6 P2, both assessors. The Trash permanent-delete confirmation does the copy well (`This cannot be undone.` over the item's title, type and trashed age, Cancel focused), then paints `Delete permanently` in ordinary body colours (rgb(225,225,225) on rgb(30,30,30), no $error) one space from a fully-styled Cancel that gets the blue focus bar. The theme defines a blocked-error/$error role for exactly this and it is unused here. The most destructive control on the surface is the least-marked one and sits one cell from the safe one. The same shape recurs in the More strip, where Move to trash ends a row of neutral actions with no separation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Delete permanently carries the theme's $error role (text or border, per the contrast rules) so it reads as destructive
- [x] #2 At least three cells separate it from Cancel
- [x] #3 Destructive entries in the More strip are visually separated from neutral ones
- [x] #4 A painted pin asserts the destructive button's colour/role differs from the neutral buttons beside it
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm the danger-style precedent (.library-media-action-danger + $error) already exists. 2. Give the confirm 'Delete permanently' the readable $error ink via an id rule that beats .library-canvas-action, and a >=3-cell margin from Cancel. 3. Mark 'Move to trash' in the More strip with the danger class + a restored left margin. 4. Painted pins at 235x52 and 100x30, red first.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The confirm 'Delete permanently' now takes `$ds-status-error-readable` (the readable error token, not the AA-failing $ds-status-error) via `#library-media-trash-delete-confirm` (id beats .library-canvas-action's $ds-text-primary), and stands >=3 cells clear of Cancel by margin (cancel margin 0, confirm margin (0,0,0,3)) rather than label padding, so the copy and the Cancel-focused default are untouched. The reader's More-strip 'Move to trash' gets `library-media-action-danger` (the Library's quiet-danger ink) plus an id-scoped `margin: 0 0 0 2` restoring the gap the more-actions `> Button {margin:0}` rule swallowed. The trash-row 'Delete forever' already carried the danger class, so it was left as-is (the finding was the CONFIRM button). Pins load the real app-CSS split sheet (a full-app harness, not a widget stub) and assert the destructive ink differs from the neutral button and the gap/margin, at both sizes, red first. Files: tldw_chatbook/css/components/_agentic_terminal.tcss (+ regenerated screen_agentic_library.tcss), tldw_chatbook/Widgets/Library/library_media_trash_canvas.py, tldw_chatbook/Widgets/Library/library_media_viewer.py, Tests/UI/test_library_media_trash.py, Tests/UI/test_library_media_reader_flow.py.
<!-- SECTION:NOTES:END -->
