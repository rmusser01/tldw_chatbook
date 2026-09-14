---
id: TASK-32557
title: >-
  Library Notes: "Add from files…" is painted "Add from" beside the grip at
  60x24
status: In Progress
assignee: []
created_date: '2026-09-13 06:48'
updated_date: '2026-09-14 16:58'
labels:
  - library
  - notes
  - critique-3
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 (dev 5fd502dbac), assessor B, compact terminal. D14. Task-32360 fixed mid-word clipping on the rail-only stage below 64 columns; task-32127's whole-word pin is at 235 wide.

**What happened.** At 60x24 the Notes list toolbar's first row paints `New  Sort: Newest  Select  Add from   s` — "Add from files…" cut to "Add from" against the grip (B 52 line 10). The footer compacts correctly and the status drops its prefix as documented. Captures: B 52.

**Cause.** INFERRED. Docs contradicted: "Nothing is ever painted as half a word."
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 At 60x24 every Notes toolbar label paints whole or is elided with an ellipsis
- [x] #2 A test at 60 columns pins the toolbar labels
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce: fresh 60x24 is clean; resizing a merged 235-wide list down to 60 paints 'New  Sort: Newest  Select  Add from   s' (the critique's exact line).
2. Trace: _effective_pane_width prefers the screen-contract pane_width, which lags one resize behind the resolved layout, so on_resize re-decides from the stale wide value and keeps the merged shape.
3. Prefer the canvas's own measured width; pane_width stays the pre-measurement fallback.
4. Pin the resize round trip through the production canvas.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Cause PROVEN, and the reported state is only reachable by a resize.

A fresh launch at 60x24 paints this toolbar correctly (two rows, 'Add from files…' whole) -- task-32360's work holds. What reproduces the critique's line exactly is narrowing an already-merged wide list: at 235 the list pane is 138 cells and the two action groups share one row; resize to 60 and the canvas kept painting that merged row inside a 50-cell pane, cutting 'Add from files…' to 'Add from' against the grip. Captured verbatim, grip included: 'New     Sort: Newest     Select       Add from   s' (wave4-caps/layout/layout-03-add-from).

`_effective_pane_width()` preferred `pane_width`, the screen's contract, over the canvas's own rendered width. That contract LAGS: the screen re-resolves the layout on a resize but the canvas keeps its previous value until a later state sync pushes a new one, so `on_resize` re-decided the toolbar's shape from the stale 138 and never flipped. Inverted: the canvas's own measured width wins, `pane_width` stays the fallback for frames that have no rendered width yet (the first frame of a visit, and the narrow route task-32360 AC#2 measured). This also fixes the other direction -- widening from 60 to 235 used to leave the toolbar in its narrow three-row shape -- and the stale 'Library notes ·' status prefix that came with it.

Live: same resize sequence now paints 'New  Sort: Newest  Select' / 'Add from files…  Export' / 'New folder' and drops the authority prefix (layout-19-resize-60-after).

Files: Widgets/Library/library_notes_canvas.py, Tests/UI/test_library_notes_w4_layout.py.
<!-- SECTION:NOTES:END -->
