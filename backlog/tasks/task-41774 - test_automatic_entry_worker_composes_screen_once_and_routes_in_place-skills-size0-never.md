---
id: TASK-41774
title: "test_automatic_entry_worker_composes_screen_once_and_routes_in_place[skills-size0] never paints the Skills row at 60x20"
status: To Do
created_date: 2026-11-18 17:00
labels:
- follow-up
- architecture-review-2026-10-25
- library
- tests
priority: low
references:
- Tests/UI/test_library_entry_compose_once.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/UI/test_library_entry_compose_once.py::test_automatic_entry_worker_composes_screen_once_and_routes_in_place[skills-size0]` fails at dev 908e9840aa with "skills terminal copy was not painted" (`:1061`). At 60x20 the painted screen never shows the Skills case's terminal row, `#library-skill-row-code-review`.

Found on 2026-11-18 while re-checking the media series' follow-ups. The test ran under a temporary private-profile marker at `-n 3`, in two rounds (load 16 and 49):
- `[skills-size0]` failed in 2 of 2 rounds;
- the test's other 17 cases passed in both rounds, including `[skills-size1]` at 170x48 and both media sizes.

Locally the test stops at `RecoveryRequired` without the marker (the file carries no profile marker), and no PR lane runs the file, so nothing has reported the failure. The file was last changed by 13cb59ee12 (TASK-36952.3.6, 2026-11-15), a receiver-only retarget of the media case. It is not known whether this is test drift at 60x20 or a product regression, nor whether it predates that series.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The notes name the cause (test drift at 60x20 or a product defect) and whether it predates the TASK-36952.3 series
- [ ] #2 The case passes 3 of 3, under the private-profile marker if the file still stops at RecoveryRequired locally
- [ ] #3 Its assertions are unchanged, unless the notes show the expectation itself was wrong
<!-- AC:END -->
