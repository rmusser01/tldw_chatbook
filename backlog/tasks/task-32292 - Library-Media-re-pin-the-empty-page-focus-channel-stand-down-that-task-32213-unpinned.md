---
id: TASK-32292
title: >-
  Library Media: re-pin the empty-page focus-channel stand-down that task-32213
  unpinned
status: To Do
assignee: []
created_date: '2026-09-10 12:40'
labels:
  - library
  - media
  - keyboard
  - critique-9
  - rider
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Qodo #2483 fixed a background recompose inside the armed entry-focus window leaving the Media list with a dead keyboard, and pinned it with a filter-MISS page — the one empty Media page that composed none of the controls the focus channel can land on. task-32213 gave that page its toolbar back, so `#library-media-type-filter` is always present and the strict "the channel cannot land at all" branch (`library_screen.py`'s `_library_media_empty_list_fallback_target() is None` leg) is no longer exercised by any test. The Conversations sibling does not cover it: that one stands down at the row-class lookup, a different branch. Evidence: task-1 review of the critique-9 media-list wave, finding 2.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] A test exercises the Media branch where `_library_media_empty_list_fallback_target()` returns `None` and asserts a background recompose inside the armed window still leaves an attached, focused widget
- [ ] `test_background_recompose_restores_focus_on_a_filtered_empty_media_list`'s docstring no longer says the strict leg is unpinned
<!-- AC:END -->
