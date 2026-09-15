---
id: TASK-32653
title: >-
  The Enhanced picker family has no footer of its own, so its modals show the host screen's chips
status: To Do
assignee: []
created_date: '2026-09-15 17:05'
labels:
  - library
  - picker
  - rider
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rider from task-32606. `EnhancedFileDialog.compose()` mirrors the base
layout by hand instead of calling it, so the `Footer` task-32606 added to
`FileSystemPickerScreen.compose` does not reach it. `EnhancedSelectDirectory`
(Personas avatar folders, vLLM model directory) therefore still shows the
HOST screen's chips through the translucent modal -- the AC#2 defect, on 32
call sites.

Adding `yield Footer()` there is not the answer on its own and was tried and
reverted: a screen-docked footer takes the bottom terminal row, and that
dialog is `height: 95%` against the vendored one's 80%. At the 60x24 its
pickers are pinned at, it pushed the character-import picker's selection
marker off the bottom and turned three existing size pins red
(`test_file_picker_action_tooltips.py` x2,
`test_file_picker_progressive.py` x1). Those pins assert what is VISIBLE at
that size, so re-baselining them would be a loosening. This needs a layout
answer for the row.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 EnhancedSelectDirectory shows its own keys instead of the host screen's while open
- [ ] #2 The 60x24 size pins in test_file_picker_action_tooltips.py stay green unmodified
<!-- AC:END -->
