---
id: TASK-32123
title: >-
  Library Notes delete receipt: Undo and Dismiss are composed off the list pane, so the only recovery path is unreachable
status: To Do
assignee: []
created_date: '2026-09-08 21:39'
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
- [ ] #1 Undo and Dismiss are visible and pressable at every width the list pane can take, including 38 columns and the compact layout
- [ ] #2 The title budget is derived from the pane width, or the buttons wrap to their own row when the pane is narrow
- [ ] #3 Covered by a geometry test at 38 columns with a 40-character title
<!-- AC:END -->
