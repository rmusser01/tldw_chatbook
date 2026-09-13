---
id: TASK-31571
title: >-
  Library media 100x30 - list view advertises esc focus rail but the rail target
  is not focusable
status: To Do
assignee: []
created_date: '2026-09-05 03:23'
labels:
  - library
  - media-ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
At 100x30 the list view's footer offers esc focus rail, but the rail element Escape targets is not focusable there, so the key does nothing (wave 4 PR B final review).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 At 100x30 Escape from the list focuses a focusable rail element, or the chip is not shown
- [ ] #2 A test at 100x30 pins it
<!-- AC:END -->
