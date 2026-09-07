---
id: TASK-31980
title: 'Library media: the permanent-delete button carries no danger affordance'
status: To Do
assignee: []
created_date: '2026-09-07 22:48'
labels:
  - library
  - media
  - ux
  - css
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #6 P2, both assessors. The Trash permanent-delete confirmation does the copy well (`This cannot be undone.` over the item's title, type and trashed age, Cancel focused), then paints `Delete permanently` in ordinary body colours (rgb(225,225,225) on rgb(30,30,30), no $error) one space from a fully-styled Cancel that gets the blue focus bar. The theme defines a blocked-error/$error role for exactly this and it is unused here. The most destructive control on the surface is the least-marked one and sits one cell from the safe one. The same shape recurs in the More strip, where Move to trash ends a row of neutral actions with no separation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Delete permanently carries the theme's $error role (text or border, per the contrast rules) so it reads as destructive
- [ ] #2 At least three cells separate it from Cancel
- [ ] #3 Destructive entries in the More strip are visually separated from neutral ones
- [ ] #4 A painted pin asserts the destructive button's colour/role differs from the neutral buttons beside it
<!-- AC:END -->
