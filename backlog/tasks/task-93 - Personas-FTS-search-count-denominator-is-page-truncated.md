---
id: TASK-93
title: Personas FTS search count denominator is page-truncated
status: To Do
assignee: []
created_date: '2026-06-11 03:23'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When the FTS path is active the 'n of m' count uses the truncated loaded-page length as m although FTS searched the full corpus. Show an accurate or unambiguous count.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Count line is accurate or explicitly unbounded when FTS is active
<!-- AC:END -->
