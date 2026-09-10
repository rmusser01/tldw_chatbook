---
id: TASK-32233
title: >-
  Library Notes: Escape is inert in the filter box and on the plain list
  although the footer promises 'esc focus rail'
status: To Do
assignee: []
created_date: '2026-09-10 14:51'
labels:
  - library
  - notes
  - keyboard
  - ux
  - critique-9
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On the Notes canvas Escape in the filter box does nothing and the next printable key is typed into it (both profiles); on the plain Notes list Escape never reaches the rail's Search Library… box and typed keys are swallowed until Enter reopens the note. The critique-8 keyboard fix (task-32051) pinned Media only (`test_library_crit8_keyboard.py::test_escape_from_a_list_filter_box_still_goes_where_the_footer_says`); `_library_list_focus_rail_target()` returns the rail box for Notes but the hop does not take effect there. Evidence: critique #9 snapshot (.impeccable/critique/2026-09-10T*__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 02374bf66a on fresh + seeded profiles, 235x52/100x30/60x24). Register row 2.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Escape in the Notes filter box blurs to the canvas and the next printable key does its canvas job
- [ ] #2 Escape on the plain Notes list (database and folder-tree layouts) focuses the rail's Search Library… box exactly as the footer says
- [ ] #3 Both cases are pinned in `Tests/UI/test_library_crit8_keyboard.py` alongside the Media pins
<!-- AC:END -->
