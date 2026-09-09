---
id: TASK-32127
title: >-
  Library Notes list pane is starved to about 38 columns at 235 wide while the work area holds about 145 empty columns
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
Observed by both assessors and the parent at 235x52: rail 34 columns, list 38, work area about 145 holding 'Select a note to edit it here.' This one geometry decision causes clipped titles ('Very long note — scaling laws d…'), the missing age column, the toolbar wrapping to three rows, hidden Move/Remove/Last import controls, the Sort strip losing its Title option and the clipped delete receipt (task-32123). 100x30 reads better than 235x52. Cause INFERRED (stylesheet not traced). Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-09T04-24-50Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev c4a7b1911f, 2026-09-08/09, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile with a 71-file Obsidian-style vault; parent re-tested every disagreement on a third profile).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With no note open, the list uses the canvas width and shows title, age and folder without clipping at 235 wide
- [ ] #2 With a note open, the list keeps at least 40 percent of the canvas or 60 columns, whichever is smaller
- [ ] #3 The toolbar fits on two rows at 235 wide with every folder and note action visible
- [ ] #4 Geometry pinned by tests at 235x52 and 100x30
<!-- AC:END -->
