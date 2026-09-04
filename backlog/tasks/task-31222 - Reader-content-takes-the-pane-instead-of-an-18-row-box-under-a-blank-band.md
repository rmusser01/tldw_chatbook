---
id: TASK-31222
title: Reader - content takes the pane instead of an 18-row box under a blank band
status: To Do
assignee: []
created_date: '2026-09-03 22:31'
labels:
  - library
  - media-ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Re-critique P1: #library-media-reader-mode-read has no CSS rule (unstyled Vertical defaults to 1fr = ~14 blank rows above the Find bar) and #library-media-viewer-content is capped max-height 18 regardless of terminal size - content gets ~1/3 of the pane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No blank band above the Find bar in the Read tab
- [ ] #2 Content height scales with the pane (long content scrolls inside; short content stays compact)
<!-- AC:END -->


## Renumbering

Renumbered from task-31204 on 2026-09-03: id collision with an older dev arrival (owner rule TASK-19601; older keeps the id).
