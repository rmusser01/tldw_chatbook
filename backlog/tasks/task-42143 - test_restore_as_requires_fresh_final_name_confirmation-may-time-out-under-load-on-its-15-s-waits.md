---
id: TASK-42143
title: "test_restore_as_requires_fresh_final_name_confirmation may time out under load on its 15 s waits"
status: To Do
assignee: []
created_date: 2026-11-20 20:06
updated_date: 2026-11-20 20:06
labels:
- follow-up
- tests
- console
- flaky
priority: low
references:
- Tests/UI/test_console_archive_recovery_flow.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the consoletests stream (round 18). `Tests/UI/test_console_archive_recovery_flow.py::test_restore_as_requires_fresh_final_name_confirmation` (line 621) failed once in a loaded local run. The file's `wait_until` (line 50) allows 15 s off Windows (60 s on Windows), polling every 30 ms, and the test makes three such waits: for the rename modal, the revised confirmation and the restored workspace. The failing run's output was not kept, so which wait timed out is not known. Two other nodes of this file are in the UI census; this one is not.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The node is run repeatedly under load (at least 10 runs, load recorded), and the failure rate and any failing wait are recorded
- [ ] #2 If it fails: the cause is named and fixed, either with a bound justified by measured durations or with a wait on a product signal, and the same repeated run passes
<!-- AC:END -->
