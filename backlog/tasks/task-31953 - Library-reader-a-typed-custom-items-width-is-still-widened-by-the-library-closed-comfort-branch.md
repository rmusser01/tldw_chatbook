---
id: TASK-31953
title: >-
  Library reader - a typed custom items width is still widened by the
  library-closed comfort branch
status: To Do
assignee: []
created_date: '2026-09-07 08:26'
labels:
  - library
  - media-ux
  - test-debt
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
H final review: test_a_typed_custom_items_width_is_obeyed_rather_than_grown's docstring overclaims. The pre-existing library-closed comfort branch (adaptive_reader_state.py ~295-303) still widens a typed 32 to 52 at width 100 in custom mode. Not PR H's bug, but the pin's name and docstring now read as a guarantee the code does not make, which is how the next reader change gets blamed for it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Either the comfort branch obeys a typed custom width, or the test's name and docstring state the branch it does not cover
- [ ] #2 The decision and its reason are recorded in the task or beside the pin
<!-- AC:END -->
