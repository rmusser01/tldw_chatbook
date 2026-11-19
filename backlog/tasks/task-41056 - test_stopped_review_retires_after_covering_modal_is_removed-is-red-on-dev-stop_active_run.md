---
id: TASK-41056
title: "test_stopped_review_retires_after_covering_modal_is_removed is red on dev: stop_active_run returns False"
status: To Do
created_date: 2026-11-15 02:03
labels:
- follow-up
- architecture-review-2026-10-25
- tests
- console
priority: low
references:
- Tests/UI/test_console_initial_hook_review.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/UI/test_console_initial_hook_review.py::test_stopped_review_retires_after_covering_modal_is_removed` fails on dev ecad404bd4 (2026-11-15) under the bootstrap profile: `controller.stop_active_run()` returns False where the test expects the stop to be accepted.

TASK-36947.5 recorded it red at its base 5ca7ec6a03 (2026-11-06), and it has stayed red. If the product is wrong, Stop does nothing while an initial hook review sits under a covering modal. The file runs on no PR lane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The task notes say whether the product or the test is wrong, and name the commit
- [ ] #2 The test passes on dev; a fixed product defect is shown red first
<!-- AC:END -->
