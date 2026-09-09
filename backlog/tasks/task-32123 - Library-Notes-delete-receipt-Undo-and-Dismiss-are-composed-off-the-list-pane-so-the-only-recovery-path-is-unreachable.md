---
id: TASK-32123
title: >-
  Library Notes delete receipt: Undo and Dismiss are composed off the list pane,
  so the only recovery path is unreachable
status: Done
assignee: []
created_date: '2026-09-08 21:39'
updated_date: '2026-09-09 06:42'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - layout
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PROVEN x3. `library_notes_canvas.py` ellipsizes the receipt title to 42 cells and lays Static + Undo + Dismiss in a non-wrapping Horizontal inside a pane about 38 columns wide; with a 20-character title the buttons are entirely absent, with a 13-character title Undo clips to 'Und' and Dismiss is gone. The guide says the receipt is the in-Library recovery action and there is no Trash browser. At 100x30 the list pane can be collapsed so the receipt is not reachable there either. Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-09T04-24-50Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev c4a7b1911f, 2026-09-08/09, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile with a 71-file Obsidian-style vault; parent re-tested every disagreement on a third profile).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Undo and Dismiss are visible and pressable at every width the list pane can take, including 38 columns and the compact layout
- [x] #2 The title budget is derived from the pane width, or the buttons wrap to their own row when the pane is narrow
- [x] #3 Covered by a geometry test at 38 columns with a 40-character title
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Failing geometry test: 38-column pane, 40-char receipt title, both buttons must have a non-zero region.\n2. Compose the receipt as a Vertical: copy row, then a button row.\n3. Live-verify the delete receipt on the power profile.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reproduced first: mounting the canvas at a 38-column pane with a 39-character receipt title put #library-notes-delete-undo at x=38..54 -- outside the pane -- and Dismiss further out still.

The receipt is now a Vertical: the "✓ deleted · <title>" copy owns one row, the two recovery actions the next, so both are inside the pane at any width the list can take. The copy takes the pane width with `text-overflow: ellipsis` instead of a fixed 42-cell budget, and the Notes canvas drops Button's 16-cell minimum (a separate rule that task-32127 needed anyway), so the two actions cost 8 cells each rather than 32 together.

AC#2 is satisfied by its second disjunct (the actions wrap to their own row); the title budget stays a constant because the canvas has no pane width at compose time, and the ellipsis rule makes the constant harmless.

Live: deleted "Groceries (not work)" on the power profile at 235x52 -- caps/03-delete-receipt.txt shows both actions painted; at 100x30 (caps/06) the same stacking holds.

Files: tldw_chatbook/Widgets/Library/library_notes_canvas.py, tldw_chatbook/css/components/_agentic_terminal.tcss (+ regenerated screen_agentic_library.tcss), Tests/UI/test_library_notes_wave_list.py.
<!-- SECTION:NOTES:END -->
