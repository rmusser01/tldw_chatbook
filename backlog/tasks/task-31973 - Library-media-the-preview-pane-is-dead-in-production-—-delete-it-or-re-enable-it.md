---
id: TASK-31973
title: >-
  Library media: the preview pane is dead in production — delete it or re-enable
  it
status: To Do
assignee: []
created_date: '2026-09-07 20:27'
labels:
  - library
  - media
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by PR O (task-31957). Every Media canvas path passes `show_preview=False` since d99fb4a9c (the Reader is the detail half), yet the pane's builder, canvas branch, two CSS tiers and pins are still maintained, and task-31957 added an analysed marker there that no user can see. Either the compact layout gets the pane back, or the dead surface goes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A written decision: re-enable the preview pane for a named layout, or delete it
- [ ] #2 If deleted: builder, canvas branch, CSS and pins are gone and the bundle is regenerated; if re-enabled: a painted pin shows it on the real screen at the chosen width
<!-- AC:END -->
