---
id: TASK-31583
title: Library Import canvas - preserve focus across panel-wide recomposes
status: To Do
assignee: []
created_date: '2026-09-05 03:24'
labels:
  - library
  - media-ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Every job tick, Clear finished, and each per-item analyze outcome recomposes the whole Import panel and drops focus (wave 4 PR D Task 3 review M-7). A focus-preserving panel repaint is a separate change from the analyze action.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Focus stays on the focused row or control across a job tick and an analyze outcome
- [ ] #2 A test pins it
<!-- AC:END -->
