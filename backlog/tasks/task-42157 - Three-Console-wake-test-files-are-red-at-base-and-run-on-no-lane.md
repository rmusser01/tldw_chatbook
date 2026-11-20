---
id: TASK-42157
title: "Three Console wake test files are red at base and run on no lane"
status: To Do
assignee: []
created_date: 2026-11-20 22:23
updated_date: 2026-11-20 22:23
labels:
- follow-up
- tests
- console
priority: low
references:
- Tests/Chat/test_console_headless_wake_invariants.py
- Tests/Chat/test_console_fleet_wake_view_mark.py
- Tests/Chat/test_console_fleet_wake_staleness.py
- Tests/conftest.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A consolerun follow-up (round 18), confirmed on fu/r18-petsdef, which does not touch these files (same command as dev, `-n 4`): 16 failed and 3 passed.
- `Tests/Chat/test_console_headless_wake_invariants.py`: 11 of 13 failed.
- `test_console_fleet_wake_view_mark.py`: 4 of 4 failed.
- `test_console_fleet_wake_staleness.py`: 1 of 2 failed.

None of the three files is in a PR or dev-push lane (no entry in `.github/workflows/*.yml` or `scripts/ui_pr_gate_census.txt`).

The visible failures are mostly harness assertions: "the wake never delivered", "harness precondition: the manual send must run", and "the wake must be IN FLIGHT before quit". One raises `RecoveryRequired: raw_source_selection_changed` directly. Consolerun attributes them to the known local RecoveryRequired trap (plain tests outside the bootstrap-profile keep list). That makes them candidates for that keep list in `Tests/conftest.py`, but the cause is not yet shown for each test.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each failing test's cause is shown (for example by running it under a temporary @private_profile_test or on the bootstrap-profile keep list); tests that fail for another reason are named
- [ ] #2 The three files pass locally at -n 4, fixed or kept by the right mechanism
- [ ] #3 The three files run on a PR or dev-push lane, with their times recorded
<!-- AC:END -->
