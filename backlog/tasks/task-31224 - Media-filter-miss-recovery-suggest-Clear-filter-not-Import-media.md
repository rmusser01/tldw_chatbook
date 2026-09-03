---
id: TASK-31224
title: 'Media filter-miss recovery - suggest Clear filter, not Import media'
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
Re-critique P2: fresh_zero never checks canvas.query so a filter miss pins 'Import media' as recovery; meanwhile #library-media-filter-clear never rendered live (unbounded Input swallows its row; no shared action class).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A zero-match ACTIVE-query page offers clearing the filter, not importing
- [ ] #2 The Clear filter control is visible whenever a query is applied
<!-- AC:END -->


## Renumbering

Renumbered from task-31206 on 2026-09-03: id collision with an older dev arrival (owner rule TASK-19601; older keeps the id).
