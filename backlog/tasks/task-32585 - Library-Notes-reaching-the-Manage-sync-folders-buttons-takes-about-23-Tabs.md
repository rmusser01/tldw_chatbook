---
id: TASK-32585
title: 'Library Notes: reaching the Manage sync folders buttons takes about 23 Tabs'
status: To Do
assignee: []
created_date: '2026-09-14 22:47'
labels:
  - library
  - notes
  - rider
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Measured by wave-4 group 3 (task-32534) and declared out of its ACs. Tab order on the Library screen is screen-wide, so from where a reader lands, the root controls in Manage sync folders are roughly 23 Tab presses away. The wave made those controls honest — each names its action in the footer now — but honest controls a keyboard user cannot reach in fewer than two dozen presses are still unreachable in practice. The Notes editor and Import once both solved this by keeping Tab inside the task and using F6 to leave (task-32246, task-32540); the roots list has no such containment.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Tab from the roots list stays within the roots task, as it does in the note editor and Import once
- [ ] #2 F6 and Escape remain the ways out
- [ ] #3 The Tab count from arrival to the first root control is measured before and after and both numbers recorded
<!-- AC:END -->
