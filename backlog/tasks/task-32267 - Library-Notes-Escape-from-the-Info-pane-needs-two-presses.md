---
id: TASK-32267
title: >-
  Library Notes: Escape from the Info pane needs two presses
status: To Do
assignee: []
created_date: '2026-09-10 18:05'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - keyboard
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Escape from the Info pane does not return on the first press; a second press does. Observed on the first-timer edit journey, at a point where the rest of the flow is a model (the footer names the focused button, the receipt names the note, Undo restores it, the empty state is written).

Peer task-32233 covers the other half of the same complaint -- Escape inert in the filter box and on the plain list although the footer promises `esc focus rail` -- and does not cover the Info pane's press count, which is this task.

Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-10T17-32-16Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev e6cb464239, 2026-09-10, dual-assessor live review at 235x52 and 100x30/60x24 on two fresh profiles and one seeded profile with a git-backed 71-file Obsidian vault fixture; the reconciling parent re-tested every disagreement on a third pair of profiles and traced each surviving cause to code). Captures under the session scratchpad `notes-crit2/{C,D,R}/caps/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One Escape press from the Info pane returns to the previous surface
- [ ] #2 Covered by a test
<!-- AC:END -->
