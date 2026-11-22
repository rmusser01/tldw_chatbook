---
id: TASK-42243
title: "test_console_fixture_ownership loses its instance lock after test_app_shutdown runs earlier on the same xdist worker"
status: To Do
assignee: []
created_date: 2026-11-22 00:17
updated_date: 2026-11-22 00:17
labels:
- follow-up
- tests
- console
priority: low
references:
- Tests/UI/test_console_fixture_ownership.py
- Tests/App/test_app_shutdown.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the testsa stream (round 19), which reported `Tests/UI/test_console_fixture_ownership.py::test_fixture_retires_captured_databases_not_replaced_attributes` red at base: `owner._instance_lock_status.handle` (line 356) is lost once `Tests/App/test_app_shutdown.py` has run on the same worker. Not reproduced from this base: the whole file passed when run alone (`-n 4`, 69 passed in a run with three other files), so the failure depends on test order within a worker. The mechanism is not named.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The test order that makes the node red is reproduced (the node and the preceding file, through the scratch wrapper) and recorded
- [ ] #2 The cause of each red node is named and fixed
- [ ] #3 The node passes in that order on a branch carrying the fix
<!-- AC:END -->
