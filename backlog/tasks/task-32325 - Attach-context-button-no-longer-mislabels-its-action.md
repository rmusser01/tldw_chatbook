---
id: TASK-32325
title: >-
  Attach context button no longer mislabels its action
status: To Do
assignee:
  - '@zcode'
created_date: '2026-09-10 12:00'
labels:
  - console
  - ux
  - rail-ux-review
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UX review B1. The control-bar button labeled 'Attach context' (console_control_bar.py:47) only opens the left rail; staging actually happens in Library, and the docs admit it parenthetically. When the rail is already open (the default) the click appears to do nothing. Either make it go where staging happens or rename it to what it does.

Filed from the 2026-09-10 Console rail UX review (review item B1).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The control no longer claims to attach anything it does not attach, and does something visible even when the left rail is already open
- [ ] #2 Composer menu entry with the same label is updated to match
- [ ] #3 User-guide rows for the control bar are updated to the new behavior/copy
- [ ] #4 Tests covering the button label/action are updated
<!-- AC:END -->
