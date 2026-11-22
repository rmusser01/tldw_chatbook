---
id: TASK-42263
title: "Fixed child-process time budgets in Backup_Recovery tests fail under load"
status: To Do
assignee: []
created_date: 2026-11-22 00:17
updated_date: 2026-11-22 00:17
labels:
- follow-up
- tests
- backup
priority: low
references:
- Tests/Backup_Recovery/test_replacement_recovery.py
- Tests/Backup_Recovery/test_first_replacement_binding.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the backuprec stream (round 19). Child-process runs in `Tests/Backup_Recovery/test_replacement_recovery.py:212` (`timeout=35`) and `Tests/Backup_Recovery/test_first_replacement_binding.py:247` (`timeout=110`) and `:281` (`timeout=90`) use fixed budgets, so they fail when the machine is loaded, though nothing is wrong.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each budget scales with measured child start-up or is replaced by a wait on the child's own completion signal, so a loaded machine does not fail it
- [ ] #2 A run of the two files at `-n 4` with the load average above 10 passes, with the load recorded
- [ ] #3 A real hang still fails, within a stated bound
<!-- AC:END -->
