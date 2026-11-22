---
id: TASK-42267
title: "No test covers readable Personal Context keys, a crash after publication, and finish"
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
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the backuprec stream (round 19). No test combines readable Personal Context keys, a process killed after publication, and `finish`. Covering it needs a test keyring that survives a process kill, because the in-process fakes are lost with the child.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A test keyring that survives a killed child process exists in the Backup_Recovery tests
- [ ] #2 A test with readable Personal Context keys, a kill after publication and `finish` passes and is red under a temporary edit that drops the keys
- [ ] #3 It runs on a PR lane
<!-- AC:END -->
