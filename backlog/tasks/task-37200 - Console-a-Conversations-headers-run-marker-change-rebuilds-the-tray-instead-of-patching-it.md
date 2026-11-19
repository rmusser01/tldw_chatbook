---
id: TASK-37200
title: >-
  Console: a Conversations header's run marker change rebuilds the tray instead
  of patching it
status: To Do
assignee: []
created_date: '2026-10-28 01:49'
labels:
  - console
  - performance
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the review of TASK-33620.15.20 (2026-10-27). The Conversations tray now patches a run's start and end in place from the row data, but a delta that changes the run marker a section or group header shows still rebuilds the whole tray: the marker is new copy appended to a title sized by its content, so writing it in place could leave the header's box the wrong size. That covers a collapsed workspace group whose rows start or end a run, and a section whose cap hides a running row. Each such rebuild re-mounts every row and costs the whole-screen reflows that TASK-36012 removed from the common case. Giving the header marker its own fixed-width slot, as the row action glyph already has, would let these deltas patch too.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A collapsed workspace group's run starting or ending updates its header marker without rebuilding the tray
- [ ] #2 A section whose cap hides a running row updates its header marker without rebuilding the tray
- [ ] #3 Each such patch lands the same geometry, paint, classes and tooltips as a fresh build of the same state, in both glyph modes
<!-- AC:END -->
