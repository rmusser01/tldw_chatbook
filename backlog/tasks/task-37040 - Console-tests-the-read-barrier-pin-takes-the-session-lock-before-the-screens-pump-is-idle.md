---
id: TASK-37040
title: >-
  Console tests: the read-barrier pin takes the session lock before the screen's
  pump is idle
status: To Do
assignee: []
updated_date: '2026-11-22 21:17'
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

2026-11-22 (round-21 filing, on the round-21 stack 77f124a740 = dev ddd3c02052 plus seven ports): the pin's `[same_session_drain]` case is now red every time, not only racy. It failed in TASK-42342's two covering runs and in two runs here, one of them with only this test's three cases and eight other nodes at `-n 4` (load 12): `--- child failure: AssertionError: released original held reader never returned; {"entered": false, "future_capture": "not_entered", ..., "released": true, "reader_returned": false}`. The held reader is never entered, so the case fails before any lock or Pilot wait this task describes. `[after_close]` and `[coalesced]` pass. The file runs in no PR lane. AC#3 cannot be met until the `same_session_drain` red has a cause and a fix.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The read-barrier pin takes the session lock only after the screen's pump is idle with nothing queued (the TASK-36611 _publication_quiet precondition or an equivalent)
- [ ] #2 With a whole sync queued for the screen's pump immediately before the settle, the pin still passes; it fails on the current settle
- [ ] #3 The pin passes repeated runs under parallel load
<!-- AC:END -->
