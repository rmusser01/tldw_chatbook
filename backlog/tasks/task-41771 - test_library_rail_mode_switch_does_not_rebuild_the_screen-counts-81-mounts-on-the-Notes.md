---
id: TASK-41771
title: "test_library_rail_mode_switch_does_not_rebuild_the_screen counts 81 mounts on the Notes switch under load, against a bound of 69"
status: To Do
created_date: 2026-11-18 16:14
labels:
- follow-up
- architecture-review-2026-10-25
- library
- tests
- flaky
priority: low
references:
- Tests/UI/test_library_phase_c_switch_residency.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/UI/test_library_phase_c_switch_residency.py::test_library_rail_mode_switch_does_not_rebuild_the_screen` (`:263`) fails under load with "notes (switch): 81 widget mounts, expected <= 69" (`_MAX_SWITCH_MOUNTS`).
- **First seen** by TASK-36952.3.3 (2026-11-15, chain over dev ecad404bd4), on dev code.
- **Re-checked** on 2026-11-18 at dev 908e9840aa under the private-profile marker at `-n 3`: it passed at load 16 and failed at load 49 with the same message.

Locally it stops at `RecoveryRequired` without the marker, and no PR lane runs the file.

The recipe's teardown probe is a different instrument. It counted 80-81 mounts for the Notes switch on most runs, even at load under 5 (TASK-36952.3.8 AC #5 data). Load may therefore not be the whole cause.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The notes name the cause of the 81-mount Notes switch: a load-dependent extra mount pass, or a second real switch path
- [ ] #2 A real second path is fixed, or the bound is re-derived with that evidence; a load effect no longer changes what the test measures
- [ ] #3 The test passes 10 of 10 at -n 4 under load, under the private-profile marker if it still stops at RecoveryRequired locally, with the load recorded
<!-- AC:END -->
