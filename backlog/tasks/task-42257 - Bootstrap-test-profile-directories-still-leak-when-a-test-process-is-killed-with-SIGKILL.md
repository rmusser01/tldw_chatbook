---
id: TASK-42257
title: "Bootstrap test-profile directories still leak when a test process is killed with SIGKILL"
status: To Do
assignee: []
created_date: 2026-11-22 00:17
updated_date: 2026-11-22 00:17
labels:
- follow-up
- tests
- tmpdir
priority: low
references:
- Tests/conftest.py
- backlog/tasks/task-42145 - tldw_test_config_-bootstrap-profile-directories-accumulate-in-TMPDIR.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the testsa stream (round 19). TASK-42145 removes the bootstrap test-profile directories at `atexit`, on SIGTERM and on SIGHUP. A process ended by SIGKILL runs none of these, so its `tldw_test_config_*` directory stays in `$TMPDIR`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A directory left by a SIGKILLed test process is removed by the next test run or by the wrapper, with the rule recorded
- [ ] #2 A test kills a child process with SIGKILL and shows the directory is cleaned afterwards
- [ ] #3 A run never removes a directory owned by a live process
<!-- AC:END -->
