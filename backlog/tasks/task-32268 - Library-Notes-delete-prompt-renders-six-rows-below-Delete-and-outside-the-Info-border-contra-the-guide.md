---
id: TASK-32268
title: >-
  Library Notes delete prompt renders six rows below Delete and outside the
  Info border, contra the guide
status: To Do
assignee: []
created_date: '2026-09-10 18:05'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Residual of task-32132, which fixed the pane-snap: the prompt now stays on Info instead of throwing the pane to Edit. What remains is that it renders six rows below the Delete button that raised it and **outside** the Info border, so the guide's claim that the confirmation "renders where Delete was pressed" with Info staying open is overstated, and the prompt is detached from the control at the moment of highest anxiety for a first-timer.

Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-10T17-32-16Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev e6cb464239, 2026-09-10, dual-assessor live review at 235x52 and 100x30/60x24 on two fresh profiles and one seeded profile with a git-backed 71-file Obsidian vault fixture; the reconciling parent re-tested every disagreement on a third pair of profiles and traced each surviving cause to code). Captures under the session scratchpad `notes-crit2/{C,D,R}/caps/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The prompt renders adjacent to the Delete control and inside the Info border
- [ ] #2 The guide's claim matches the live surface
- [ ] #3 Covered by a test asserting the prompt's position relative to the control that raised it
<!-- AC:END -->
