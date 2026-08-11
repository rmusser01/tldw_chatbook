---
id: TASK-15110
title: >-
  Console context rail: Conversations starts below the fold with all sections open
status: To Do
assignee: []
created_date: '2026-08-11 04:00'
labels:
  - console
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Measured while repairing task-14920, and raised rather than guessed at.

TASK-14810 split the Console context rail into sections. The split itself is correct and the rows are genuinely reachable — but with all three sections expanded by default, `#console-left-rail-body` has a virtual height of **99 rows against a 29-row viewport** at 160x48: the Conversations section body starts at y=45 and its first row sits at y=70, roughly 20 rows below the fold. Reaching it needs a scroll the user has no cue to perform.

This surfaced because 12 tests began failing with `textual.pilot.OutOfBounds` — `pilot.click` addresses screen coordinates, so a target below the fold reports a coordinate error rather than the layout fact that caused it. Those tests were repaired by scrolling first (the honest test fix, since the rows do work once visible), which means **nothing now fails if the rail grows further**: the shipped tests for TASK-14810 assert section order and independent collapse, never on-screen reachability.

So this is a discoverability question for the owner, not a regression: should all three sections default to open when the third lands off-screen, should the rail remember collapse state, or should sections default collapsed below some height? Whatever is chosen, a test that pins reachability would stop the next growth from being invisible.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 An owner ruling is recorded on the default expansion of the rail's sections at common terminal sizes
- [ ] #2 The chosen behaviour is implemented, and the first row of every default-visible section is reachable without scrolling at a supported size
- [ ] #3 A test pins on-screen reachability (not just section order and collapse), so future rail growth fails loudly instead of silently pushing content off the fold
<!-- AC:END -->
