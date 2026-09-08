---
id: TASK-31962
title: Library media - three spellings of the backing-id int coercion
status: Done
assignee: []
created_date: '2026-09-07 08:27'
updated_date: '2026-09-07 20:20'
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
- [x] #1 One helper owns the coercion and all three call sites use it
- [x] #2 Its pins cover the shapes each old spelling handled: a bare int, a prefixed display id, and an unparseable value
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. One helper for the backing-id coercion; pins for a bare int, a prefixed display id and an unparseable value. 2. Route the three call sites through it.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
One helper in `library_media_state.py` owns the coercion; `_library_media_int_backing_id`, the review-selected handler's inline split and `_review_cursor_for_display`'s try/except all use it. Nine parametrized shapes plus two handler-level pins. Non-positive ids are refused (`local:media:0` cannot reach the selection worker; `validate_media_browse_items` already requires `backing_media_id >= 1`), pinned rather than silent.
<!-- SECTION:NOTES:END -->
