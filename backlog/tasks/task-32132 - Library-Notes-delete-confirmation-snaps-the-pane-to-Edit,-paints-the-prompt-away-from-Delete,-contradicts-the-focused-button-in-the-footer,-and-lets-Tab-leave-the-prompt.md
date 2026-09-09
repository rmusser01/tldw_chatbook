---
id: TASK-32132
title: >-
  Library Notes delete confirmation snaps the pane to Edit, paints the prompt away from Delete, contradicts the focused button in the footer, and lets Tab leave the prompt
status: To Do
assignee: []
created_date: '2026-09-08 21:39'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - keyboard
  - copy
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Observed by the design assessor and reproduced by the parent: pressing Delete in Info switches the pane back to Edit and paints 'Delete this note?' below the body editor, 14 rows lower; the footer reads 'enter confirm delete | esc cancel delete' while Cancel holds focus, and Enter cancels; eight Tabs walk focus out of the prompt into a pane grip with the prompt still open. Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-09T04-24-50Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev c4a7b1911f, 2026-09-08/09, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile with a 71-file Obsidian-style vault; parent re-tested every disagreement on a third profile).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The confirmation renders in the Info Danger section where Delete was pressed, without changing mode
- [ ] #2 The footer describes the focused button (Enter cancels while Cancel is focused)
- [ ] #3 Tab and Shift+Tab cycle between Cancel and Delete while the prompt is open
- [ ] #4 Covered by tests
<!-- AC:END -->
