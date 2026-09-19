---
id: TASK-32588
title: >-
  Library Notes: the Import once files branch joins up to three bare basenames
  on one line at the 48-column floor
status: To Do
assignee: []
created_date: '2026-09-14 22:48'
labels:
  - library
  - notes
  - rider
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Disputed and filed as a rider by wave-4 group 7 rather than changed inside task-32554, because it is a different budget question with its own prior ruling: the files branch really does join up to three bare basenames on one line at the un-widened 48 floor, and task-32122 review round 2 deliberately chose head-truncate over elide_path_middle for them (_bounded_source_name's docstring records that ruling). The result is still hard to read — three truncated names on one line with no separator that survives truncation — and 32554 fixed the neighbouring elision problem for the folder branch only. Worth deciding on its own terms.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The files-branch confirmation line at 48 columns is legible: each chosen file is identifiable from what is painted
- [ ] #2 Whatever is chosen is reconciled with task-32122 round 2's head-truncate ruling, either by superseding it with a reason or by keeping it and solving the legibility another way
- [ ] #3 Measured at 48 and 60 columns with a capture each, with one, two and three files chosen
<!-- AC:END -->
