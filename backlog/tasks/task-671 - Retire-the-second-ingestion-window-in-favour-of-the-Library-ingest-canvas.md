---
id: TASK-671
title: Retire the second ingestion window in favour of the Library ingest canvas
status: To Do
assignee: []
created_date: '2026-07-26 03:27'
labels:
  - ingest
  - cleanup
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The app ships two independently built ingestion interfaces with different layouts, different defaults and different selection models, reachable from different places. The second one adds files to a batch with no way to remove one, ignores a repeat click, and applies a single title to every file in the batch. Maintaining two answers to the same job doubles the bug surface and guarantees they drift.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Import sources opens the Library ingest canvas
- [ ] #2 No route or button reaches the retired window
- [ ] #3 Any capability the retired window had that the canvas lacked is available in the canvas
- [ ] #4 The retired window and its now-unused event handlers are deleted
- [ ] #5 The full test suite passes with the window removed
<!-- AC:END -->
