---
id: TASK-37040
title: >-
  Console tests: the read-barrier pin takes the session lock before the screen's
  pump is idle
status: To Do
assignee: []
created_date: '2026-10-27 16:20'
labels:
  - console
  - tests
  - reliability
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-36611 found that a Console test which takes the tab strip's session lock right after a 'no sync in progress' settle can catch a whole sync still queued for the screen's own pump (the attach reconciliation's view start). The screen's pump then parks behind the test's lock, and any Pilot wait or click the test makes while holding it times out. Tests/UI/test_console_tab_publication_read_barrier.py::test_actual_close_publishes_real_tabs_before_stock_read_release takes the surface lock after the same weak settle and then clicks the close button while holding it. It passed 3 of 3 on 2026-10-27, but nothing it waits for rules out that queued sync.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The read-barrier pin takes the session lock only after the screen's pump is idle with nothing queued (the TASK-36611 _publication_quiet precondition or an equivalent)
- [ ] #2 With a whole sync queued for the screen's pump immediately before the settle, the pin still passes; it fails on the current settle
- [ ] #3 The pin passes repeated runs under parallel load
<!-- AC:END -->
