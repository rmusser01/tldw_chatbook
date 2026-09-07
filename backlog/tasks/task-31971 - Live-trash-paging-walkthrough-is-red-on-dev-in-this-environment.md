---
id: TASK-31971
title: Live trash-paging walkthrough is red on dev in this environment
status: To Do
assignee: []
created_date: '2026-09-07 20:27'
labels:
  - library
  - media
  - tests
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during PR N's final review census. `Tests/Live/test_library_media_trash_paging_closeout.py::test_live_real_database_media_trash_walkthrough` is red on dev 45706ae57 locally: the Trash page-2 load raises RuntimeError and the walk's wait times out. It is a collected default-suite test that is not in task-31249's census, so its red hides in whole-suite noise.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The walkthrough passes on dev locally, or is marked with the reason it cannot run here and excluded from the default collection
- [ ] #2 The root cause of the page-2 RuntimeError is named in this task
<!-- AC:END -->
