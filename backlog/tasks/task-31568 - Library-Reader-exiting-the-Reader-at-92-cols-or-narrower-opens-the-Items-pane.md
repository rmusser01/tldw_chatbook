---
id: TASK-31568
title: >-
  Library Reader - exiting the Reader at 92 cols or narrower opens the Items
  pane
status: To Do
assignee: []
created_date: '2026-09-05 03:22'
labels:
  - library
  - media-ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
At 92 columns or narrower both panes are collapsed; exiting the Reader flips the view to list while the Reader keeps painting and ] and [ go dead. Wave 4 PR B narrowed the user-guide claims instead of fixing it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Escape or Back at 92 cols or narrower opens the Items pane with the current row focused
- [ ] #2 ] and [ work after the exit, or the footer says why they do not
- [ ] #3 The user guide's narrow-layout promises are restored to match
<!-- AC:END -->
