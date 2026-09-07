---
id: TASK-31979
title: >-
  Library media: the wide layout truncates a long title while the reader pane
  sits empty
status: To Do
assignee: []
created_date: '2026-09-07 22:48'
labels:
  - library
  - media
  - ux
  - layout
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #6 P2, both assessors. At 235x52 a 98-character title is cut to 47 characters in a fixed 52-column list pane while roughly 120 columns of reader pane hold the single sentence `Select a media item to read it here.`. At 100x30 the same title wraps across three fully readable lines. The wide layout is the one that loses information: the two-pane split reserves reader width that nothing uses when no item is open.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 When no media item is open, the list pane is allowed enough width to show substantially more of a long title (or the title wraps rather than truncating)
- [ ] #2 Opening an item restores the reader pane's width
- [ ] #3 A pin asserts a long title paints more characters at 235x52 with no item open than the current fixed 52-column pane allows
<!-- AC:END -->
