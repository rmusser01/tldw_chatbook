---
id: TASK-41515
title: "Backup_Recovery tests that are load-sensitive: control_records enrollment wait, the later-snapshot fixtures, restore_data_groups and the restart journey's 5 s waits"
status: To Do
assignee: []
created_date: 2026-11-17 05:27
labels:
- follow-up
- tests
- backup
priority: medium
references:
- Tests/Backup_Recovery/test_builtin_later_snapshot.py
- Tests/Backup_Recovery/test_restore_data_groups.py
- Tests/ProductionApp/test_backup_restore_end_to_end.py
dependencies:
- TASK-41116
- TASK-41117
- TASK-41118
- TASK-41305
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the bktests stream (round 15). Still open after TASK-41116, TASK-41117, TASK-41118 and TASK-41305:
- `control_records._enrollment`'s 0.1 s wait refused once under load;
- `test_builtin_later_snapshot.py` and the `native_package` fixture: per-test limits need a measurement at low load (about 40) with the 300 s limit lifted (TASK-41117 left this as a stated follow-up);
- `test_restore_data_groups.py`: 4 cases hit 300 s at load about 110 in the backup-owner-suites-dev lane; the first CI runs should be read before moving limits;
- the end-to-end restart journey's fixed 5 s waits (`TimeoutError`, `wait(timeout=5)`) block four of TASK-41305's cases at `-n 4`, load 60-70.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each wait named uses a condition wait or a measured limit, with the load of the measurement recorded
- [ ] #2 `test_builtin_later_snapshot.py` has measured per-test `timeout` marks, or `complete_builtin_case` is cheaper
- [ ] #3 The four restart-journey cases pass at `-n 4`, load 60-70
- [ ] #4 The first CI runs of `test_restore_data_groups.py` are read and the outcome is recorded
<!-- AC:END -->
