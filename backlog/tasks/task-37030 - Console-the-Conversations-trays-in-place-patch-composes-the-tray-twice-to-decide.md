---
id: TASK-37030
title: >-
  Console: the Conversations tray's in-place patch composes the tray twice to
  decide
status: To Do
assignee: []
created_date: '2026-10-26 14:32'
labels:
  - console
  - performance
dependencies:
  - TASK-36012
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-36012 lets the Conversations tray repaint a run's start and end in place: on a delta confined to run markers, attention facts, icons and colours it composes the tray twice off the DOM (the state shown and the next one), diffs the two, and patches the mounted rows when no box moves. Those two composes cost about what the rebuild they replace did (with the app's stylesheets, a running-marker sync measured 81 ms of CPU against dev's 81 ms at 12 rows and 211 ms against 205 ms at 40 rows), so the patch saves the whole-screen reflows a rebuild triggers but no CPU inside the tray, and all of the cost now blocks inside sync_state. The TASK-36012 review measured most of a compose's cost in the stylesheet applies that add_class and set_class run on widgets that are not yet mounted. Recording each patchable node's compose-time values when the tray composes would let the patch skip the compose of the state already shown, roughly halving the work on every send's start and end.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A run marker that the tray patches in place runs at most one compose of the tray
- [ ] #2 The patch still lands the same geometry, paint, classes and tooltips as a rebuild (the TASK-36012 A/B stays green)
- [ ] #3 CPU for a patched running-marker sync, measured with process_time at 12 and 40 rows, falls below the dev baseline
<!-- AC:END -->
