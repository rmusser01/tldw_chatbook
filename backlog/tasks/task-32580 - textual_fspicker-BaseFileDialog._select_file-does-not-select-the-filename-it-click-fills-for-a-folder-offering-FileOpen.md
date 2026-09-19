---
id: TASK-32580
title: >-
  textual_fspicker: BaseFileDialog._select_file does not select the filename it
  click-fills for a folder-offering FileOpen
status: To Do
assignee: []
created_date: '2026-09-14 22:46'
labels:
  - library
  - rider
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by wave-4 group 7 while delivering task-32540. Clicking a file row fills the File name field with that name, but does not select it — so the next keystroke appends to the filled name instead of replacing it. FileSave has selected-on-fill since task-1479; a folder-offering FileOpen does not, and the two dialogs are otherwise the same control. One line brings them into agreement and improves both.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Clicking a file row in a folder-offering FileOpen fills AND selects the filename, as FileSave has since task-1479
- [ ] #2 Typing immediately after the click replaces the filled name rather than appending to it
- [ ] #3 Pinned by a test that delivers a real click and then a keystroke, in both FileOpen and FileSave
<!-- AC:END -->
