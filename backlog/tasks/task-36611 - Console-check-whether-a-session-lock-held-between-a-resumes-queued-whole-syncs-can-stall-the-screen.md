---
id: TASK-36611
title: >-
  Console: check whether a session lock held between a resume's queued whole
  syncs can stall the screen
status: To Do
assignee: []
created_date: '2026-10-24 07:50'
labels:
  - console
  - reliability
  - ux-review-2026-09-29
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-33620.15.4 changed `Tests/UI/test_console_tab_publication_maintenance.py::test_admitted_real_tab_publication_rechecks_replaced_controller` to wait for idle across three drained pauses. The test took the session lock while a resume's whole sync was still queued, and Pilot's wait for pending messages timed out (`WaitForScreenTimeout`). The test is flaky on dev: one review saw it fail 1 run in 4, and on 2026-10-24 it failed 4 of 4 on dev under load. With dev's version of the test, the branch failed every time. The branch's fix changed only the test.

Nobody has checked whether production can reach the same state, where a session lock held between a resume's queued whole syncs blocks the Console screen from processing its messages. If it can, a user would see the Console stop responding until the lock is released.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Why the test's wait timed out is diagnosed and recorded in the task notes
- [ ] #2 A deterministic test either reproduces the stall through production code paths or shows that no production caller holds the session lock across a resume's queued whole syncs
- [ ] #3 If the stall is reproducible, the Console keeps processing messages while the lock is held, and the test fails on the base build
<!-- AC:END -->
