---
id: TASK-32648
title: >-
  Library Notes: Tab from the Folder files search field leaks into the Library rail instead of the Files tree
status: To Do
assignee: []
created_date: '2026-09-15 17:05'
labels:
  - library
  - notes
  - critique-4
  - rider
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rider from task-32606. Observed live at 235x52 (twice) and 100x30 on a
scratch power profile: pressing `/` focuses the folder navigator's
"File contents..." field, and Tab from there lands on a Library rail row, not
on the `#file-notes-tree` immediately below it. Arrow keys then drive the
rail. This is what stopped task-32606 proving the last leg of its AC#3 live;
that leg is pinned headless instead.

INFERRED: not traced to a Tab region. Probably the same class of defect
task-32246 / task-32540 AC#3 fixed for the import stepper, where
`_LIBRARY_TAB_REGION` spanned `#screen-content` and Tab leaked out of the
canvas.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Tab from the Folder files search field reaches the Files tree
- [ ] #2 Tab does not leave the Folder files canvas while that mode is active
- [ ] #3 A keyboard-only test walks search field -> tree -> open a file
<!-- AC:END -->
