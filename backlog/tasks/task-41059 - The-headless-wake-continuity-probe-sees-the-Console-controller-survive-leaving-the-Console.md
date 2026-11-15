---
id: TASK-41059
title: "The headless-wake continuity probe sees the Console controller survive leaving the Console"
status: To Do
created_date: 2026-11-15 02:03
labels:
- follow-up
- architecture-review-2026-10-25
- tests
- console
priority: low
references:
- Tests/UI/test_probe_headless_wake_p1_continuity.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/UI/test_probe_headless_wake_p1_continuity.py::test_probe_p1_db_rows_written_while_console_is_unmounted` fails on dev ecad404bd4 (2026-11-15) under the bootstrap profile: "leaving Console must shut the controller down". The test's shutdown event is never set after the Console screen is left.

TASK-36947.4's review (2026-11-08) recorded it red at its base, alone and in bulk. If the product is wrong, a Console controller keeps running after its screen is gone. If the test is wrong, the probe no longer measures what its name says. The file runs on no PR lane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The task notes say whether the product or the test is wrong, and name the commit
- [ ] #2 The test passes on dev; a fixed product defect is shown red first
<!-- AC:END -->
