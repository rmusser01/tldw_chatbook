---
id: TASK-41909
title: "Four Library and Schedules UI tests flake under load"
status: To Do
assignee: []
created_date: 2026-11-18 20:31
updated_date: 2026-11-18 20:31
labels:
- follow-up
- tests
priority: low
references:
- Tests/UI/test_library_media_render_fixes.py
- Tests/UI/test_schedules_workbench.py
- Tests/Chat/test_console_trace_maintenance_parking.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Seen in round-16 runs, unchanged by round 16: `test_more_reads_as_an_open_disclosure_while_it_is_open` ("More ▴" still painted after close), `test_programmatic_media_row_restore_keeps_reader_until_user_keys` (about 10-15% under load, Media owners not settling), the Schedules definition Runs-on nodes (`test_definition_runs_on_retry_button_rebegins_a_failed_transfer` and siblings, about 1 in 3-5), and `test_a_signal_woken_real_pass_opens_only_its_write_transaction` (once in a serial recovered-suites run).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each flake's cause is named, as timing or as a defect
- [ ] #2 Each passes 10 of 10 at `-n 4` with the load recorded, or its defect is fixed with a red-before test
<!-- AC:END -->
