---
id: TASK-31633
title: >-
  Library media wide layout - let the list grow with width and stop More
  displacing the body
status: To Do
assignee: []
created_date: '2026-09-05 06:18'
labels:
  - library
  - media-ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #5 P1: at 235x52 the Items list is 38 cells and truncates a 98-character title while at 100x30 it is about 47 cells and fits; two 5-cell gutters flank it; the Reader lays 83 characters into a 145-cell frame; each item costs three rows; opening More pushes the tab row and the body down about 19 rows. The Items-pane floor was set for the collapse case and never told to grow.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 At 235x52 the list column is at least as wide as at 100x30 and a 98-character title fits or truncates later
- [ ] #2 No 5-cell dead gutter remains between rail, list and Reader
- [ ] #3 More renders without displacing the Reader body
- [ ] #4 Painted tests pin the widths at both sizes
<!-- AC:END -->
