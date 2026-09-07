---
id: TASK-31578
title: Library media search UI test pins the fake service instead of the keyword leg
status: To Do
assignee: []
created_date: '2026-09-05 03:24'
labels:
  - library
  - media-ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The UI test for the Title/keyword filter asserts against the fake media service, so a regression in the real MediaDatabase keyword leg would not fail it (wave 4 PR C review).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One integration test drives the Library search through a real MediaDatabase and asserts a keyword-only match
<!-- AC:END -->
