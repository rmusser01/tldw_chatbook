---
id: TASK-32102
title: >-
  Library structural waits: follow-ups from the task-32055 review (timer
  retention, File Notes Cancel timing, narrow back toggle, cancelled-import
  refresh)
status: To Do
assignee: []
created_date: '2026-09-08 22:42'
labels:
  - library
  - file-notes
  - skills
  - cleanup
  - critique-8
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the task-32055 reviews (PR #2524): the patience `set_timer` in `_begin_library_structural_wait` is neither retained nor stopped; the File Notes Cancel button is revealed at t=0 while the status line mentions Cancel only after 3 s; `#file-notes-back` (narrow-view toggle) does not cancel a running folder change (the 30 s deadline is the only backstop); a cancelled skill import leaves `refresh_sources=False` although its copy says to check the skills list; `except asyncio.CancelledError: if not wait.cancelled: raise` tests the wait flag rather than `change.cancelled()`; the wait branch of `_update_root_surface` skips the -warning/-offline resets; the cancelled copy is written to a canvas being torn down on the exit-triggered path. Rider from the critique-8 fix wave reviews (plan Docs/superpowers/plans/2026-09-08-library-crit8-wave.md; wave PRs #2519 #2523 #2524 #2525 #2528 #2531).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each of the seven items is fixed or closed with a recorded reason
- [ ] #2 The File Notes Cancel affordance and its copy appear at the same moment
<!-- AC:END -->
