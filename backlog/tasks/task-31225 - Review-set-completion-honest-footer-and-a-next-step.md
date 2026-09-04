---
id: TASK-31225
title: Review-set completion - honest footer and a next step
status: To Do
assignee: []
created_date: '2026-09-03 22:31'
labels:
  - library
  - media-ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Re-critique P2: after 'All N reviewed' the footer keeps advertising '] next in set' (now a silent no-op, violating the task-28005 honest-footer rule) and offers no next step. Riders: the storage-unavailable notice ships error twice/warning once (normalize); investigate B's unattributed Space-in-empty-select-mode canvas blank at 100x30 (pane-grip hypothesis).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A completed set's footer stops advertising ] and keeps R (and m for un-marking)
- [ ] #2 Completion offers a next step
<!-- AC:END -->


## Renumbering

Renumbered from task-31207 on 2026-09-03: id collision with an older dev arrival (owner rule TASK-19601; older keeps the id).
