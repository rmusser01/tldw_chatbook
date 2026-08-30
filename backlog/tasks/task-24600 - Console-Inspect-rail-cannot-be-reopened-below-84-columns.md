---
id: TASK-24600
title: Console Inspect rail cannot be reopened below 84 columns
status: To Do
assignee: []
created_date: '2026-08-30 00:53'
labels:
  - console
  - ux
  - inspector
  - critique-2026-08-29
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Below the 84-column single-pane threshold both rail handles hide. A rail that is explicitly open at that width can still be collapsed by its own header button, after which nothing on screen references the Inspector: no handle, no status chip route, no keyboard binding, no command-palette entry. The only observed recovery is resizing the terminal. Small-terminal users lose the Inspector for the session.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Collapsing the Inspect rail below 84 columns always leaves a visible, activatable affordance that reopens it
- [ ] #2 The reopen affordance is reachable by keyboard alone, not only by mouse
- [ ] #3 A regression test drives collapse then reopen at 80x24 and asserts the rail is displayed again
<!-- AC:END -->
