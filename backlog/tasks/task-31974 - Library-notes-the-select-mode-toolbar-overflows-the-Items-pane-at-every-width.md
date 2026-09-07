---
id: TASK-31974
title: 'Library notes: the select-mode toolbar overflows the Items pane at every width'
status: To Do
assignee: []
created_date: '2026-09-07 20:27'
labels:
  - library
  - notes
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by PR O (task-31959). Notes' select-mode action row never fits the 36-cell Items pane, so `Clear` and `Export selected` are never painted on the real screen; the 31959 pin has to drive a bare canvas with the CSS restated. Rework the row the way PR F put Media's Done on its own row. While there: `library_notes_canvas.py` (~1514) hard-codes `align=True` instead of reading the button's `_library_disabled_marker_align` like the shared patcher does.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 In select mode on Notes, every action (Done, All, Clear, Export selected) is painted inside the Items pane at 235×52 and 100×30 (painted pins over the real screen)
- [ ] #2 The Notes column pin drives the real screen, not a bare canvas
- [ ] #3 The Notes canvas reads the align flag from the button rather than hard-coding it
<!-- AC:END -->
