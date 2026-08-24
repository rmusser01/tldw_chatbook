---
id: TASK-21243
title: >-
  Export from the Notes canvas still takes the whole-screen recompose arm
status: To Do
assignee: []
created_date: '2026-08-23'
labels:
  - library
  - performance
dependencies: []
priority: medium
---

## Description

Source: close-out of the 2026-08-22 holistic performance review burn-down; the fifth Minor
from the TASK-21116 review. Split from TASK-21242 (the four correctness gaps) because this is
the same class of work TASK-21116 itself did — moving another site onto the seam.

The holistic review counted **~105 whole-screen `refresh(recompose=True)` sites** in
`library_screen.py` (99 on `self.`), on a screen that grew from 26k to 34.8k lines, and
TASK-21116 converted the confirmed-hot per-click ones to the canvas-scoped seam. Export
initiated from the Notes canvas still routinely takes the structural whole-screen arm
(`library_screen.py:29616` reaching `:8556` at the time of the review) rather than a
canvas-scoped sync — so a routine user action still recomposes the largest screen in the app.

## Acceptance Criteria

- [ ] Export initiated from the Notes canvas performs a canvas-scoped sync, not a whole-screen recompose
- [ ] A test asserts the recompose count for that action and fails if the whole-screen arm is taken
- [ ] Export behaviour and the resulting UI state are unchanged
- [ ] The whole-screen recompose site ratchet moves down and stays green
