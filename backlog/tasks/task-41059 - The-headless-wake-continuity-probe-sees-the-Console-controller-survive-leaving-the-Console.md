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
`Tests/UI/test_probe_headless_wake_p1_continuity.py::test_probe_p1_db_rows_written_while_console_is_unmounted` fails on dev ecad404bd4 (2026-11-15) under the bootstrap profile: "leaving Console must shut the controller down". A plain local run stops earlier at RecoveryRequired: raw_source_selection_changed (TASK-32908); these results come from adding the bootstrap_profile marker that Tests/conftest.py honours (for example via a -p plugin) and running each node alone. The probe can also create a red of its own (`Tests/Chat/test_console_skill_script_confirm.py::test_request_skill_script_confirm_parks_for_a_non_active_session` passes plain and fails under it), so check a node that fails only under the probe before treating it as real. The test's shutdown event is never set after the Console screen is left.

TASK-36947.4's review (2026-11-08) recorded it red at its base, alone and in bulk. If the product is wrong, a Console controller keeps running after its screen is gone. If the test is wrong, the probe no longer measures what its name says. The file runs on no PR lane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The task notes say whether the product or the test is wrong, and name the commit
- [ ] #2 The test passes on dev; a fixed product defect is shown red first
<!-- AC:END -->
