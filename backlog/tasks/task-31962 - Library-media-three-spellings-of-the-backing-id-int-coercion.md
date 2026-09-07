---
id: TASK-31962
title: Library media - three spellings of the backing-id int coercion
status: To Do
assignee: []
created_date: '2026-09-07 08:27'
labels:
  - library
  - media-ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
J final review M3: library_screen.py now coerces a display id to its int backing id three ways - _library_media_int_backing_id, the review-selected handler's inline split, and _review_cursor_for_display's try/except. Three parsers for one id format is how the next change to the id shape half-lands.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One helper owns the coercion and all three call sites use it
- [ ] #2 Its pins cover the shapes each old spelling handled: a bare int, a prefixed display id, and an unparseable value
<!-- AC:END -->
