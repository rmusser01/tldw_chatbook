---
id: TASK-32259
title: >-
  Library Notes: four surfaces put their primary action 20 to 38 rows below
  the content it acts on
status: To Do
assignee: []
created_date: '2026-09-10 18:05'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - layout
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The design assessor's structural finding, and the reason the specificity verdict reads "authored in the copy and the state model; category-interchangeable in the spatial design". Adjacent to peer task-32217 (no density rule across Library canvases), but these four instances are specific to Notes:

- Import once's "Check selection" at row 49 under a selection summary at rows 7-11;
- the Folder-files empty state at rows 7-9 with 47 blank rows under it (the full-screen half of this overlaps peer task-32211);
- the Markdown preview closing its box at row 33 with 14 blank rows beneath (task-32249);
- the review pane elided to `. ke...` while the Notes list it is not about keeps its share of the width (task-32250).

The brand words are "cyberpunk, efficient, effective"; half of most of these screens is empty. The one place the layout was fixed -- the notes list itself, now about 132 of 235 columns with ages and a complete two-row toolbar (task-32127) -- proves the rest is fixable.

Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-10T17-32-16Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev e6cb464239, 2026-09-10, dual-assessor live review at 235x52 and 100x30/60x24 on two fresh profiles and one seeded profile with a git-backed 71-file Obsidian vault fixture; the reconciling parent re-tested every disagreement on a third pair of profiles and traced each surviving cause to code). Captures under the session scratchpad `notes-crit2/{C,D,R}/caps/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Import once's primary action renders adjacent to the selection summary it acts on, not at the pane floor
- [ ] #2 A full-canvas task (import review, sync setup, export) owns the pane width while it is the task in hand
- [ ] #3 No Notes surface renders its primary action more than a screenful below the content it acts on at 235x52
- [ ] #4 Covered by a test asserting the action's row distance from its content on the Import once selection screen
<!-- AC:END -->
