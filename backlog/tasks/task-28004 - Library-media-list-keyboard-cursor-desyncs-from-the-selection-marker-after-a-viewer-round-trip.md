---
id: TASK-28004
title: >-
  Library media list - keyboard cursor desyncs from the selection marker after a
  viewer round-trip
status: To Do
assignee: []
created_date: '2026-09-02 04:10'
labels:
  - library
  - bug
dependencies: []
references:
  - >-
    .impeccable/critique/2026-09-02T04-00-36Z__tldw-chatbook-ui-screens-library-screen-py.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live-reproduced twice: returning from the viewer leaves the visible marker on the item just read while keyboard focus is armed to the FIRST row (library_screen.py:6586, task-2856 AC1, _arm_library_list_entry_focus -> rows.first()). Arrow keys then move an invisible cursor and Enter opens a different row than the visibly marked one (Up from marked talk1 opened talk3). Fix direction: make _focus_library_list_entry prefer the row whose media_id matches _selected_media_id before falling back to first - the select-mode branch at library_screen.py:6575-6585 is the precedent. Grep for pinning tests on task-2856 AC1 before changing the focus target.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After returning from the viewer, the focused row and the visibly marked row are the same row
- [ ] #2 Enter always opens the visibly marked row
- [ ] #3 task-2856 focus tests updated or extended, plus a pinning test for the round-trip
<!-- AC:END -->
