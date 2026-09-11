---
id: TASK-32383
title: 'Library Media Trash: a Trash filtered to zero matches still says it is empty'
status: To Do
assignee: []
created_date: '2026-09-11 10:30'
labels:
  - library
  - media
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filtering Trash down to no matches renders the source-is-empty message ("Trash is empty...") instead of naming the filter that hid everything, so a user who cannot find a deleted item is told it does not exist. This is the same defect task-32352 AC#2 fixed for Collections, on a surface that fix did not reach; the Library-wide empty-state rule already distinguishes "this source has nothing" from "your filter matched nothing" and offers the reset action for the latter.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Trash filtered to zero matches names the active filter and offers the reset action that restores the unfiltered page
- [ ] #2 Trash with genuinely no deleted items keeps its own empty message
- [ ] #3 Both states are covered by tests in the Trash's own test file
<!-- AC:END -->
