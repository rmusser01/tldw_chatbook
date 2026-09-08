---
id: TASK-32044
title: >-
  Library media: a regional-indicator flag keyword miscounts by two cells and
  overpaints the row frame
status: To Do
assignee: []
created_date: '2026-09-08 14:36'
labels:
  - library
  - media
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #7 P2 (the visible consequence of task-31955's stated flag-pair ceiling). A keyword containing a regional-indicator flag emoji (e.g. the pair rendered as one flag) is width-miscounted: the row frame drifts +2 cells (border at column 237 vs 235) and the reader header overpaints (e.g. 'NoNo analysis yet.'). task-31955 cut keyword reasons by display cells with rich.cells but documented that regional-indicator pairs are not fully handled; this is that ceiling showing a real frame break.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A keyword reason containing a regional-indicator flag emoji does not drift the row frame width or overpaint neighbouring panes at 235x52 or 100x30
- [ ] #2 The cut is flag-pair-aware (a UAX #29 segmenter, or the reason refuses to paint a half-flag / cuts before the pair) rather than mis-measuring the pair
- [ ] #3 A pin paints a flag-keyword reason and asserts the row frame stays aligned
<!-- AC:END -->
