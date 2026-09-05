---
id: TASK-31567
title: Library Reader - restore focus after any Reader recompose
status: To Do
assignee: []
created_date: '2026-09-05 03:22'
labels:
  - library
  - media-ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After any recompose of the adaptive Reader shell, focus falls through to the pane grip. This is why Space collapsed the pane in wave 4 PR B Task 1 and why the retired grip end-caps kept reappearing; every fix so far patched one caller. A general restore-focus-after-recompose seam is needed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After a Reader recompose, focus returns to the widget that held it (row, content, Find input) rather than a pane grip
- [ ] #2 Space on a focused Items row never collapses a pane
- [ ] #3 Painted tests at 235x52 and 100x30 cover the row, content and Find cases
<!-- AC:END -->
