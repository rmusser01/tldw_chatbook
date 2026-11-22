---
id: TASK-42252
title: "Four Console tests still hit the 300 s timeout after TASK-40638"
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
- Tests/Chat/test_console_turn_execution_context.py
- Tests/Chat/test_console_voice_preflight.py
- Tests/Chat/test_console_hook_preparation_lifetime.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the consoleb stream (TASK-42156, round 19). Not covered by TASK-40638: `Tests/Chat/test_console_turn_execution_context.py` (2 tests), `Tests/Chat/test_console_voice_preflight.py::test_readiness_cancellation_propagates` (line 174) and `Tests/Chat/test_console_hook_preparation_lifetime.py::test_original_v2_configuration_outlives_close_and_repeated_cancellation` (line 188) each run to the 300 s timeout.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The cause of each red node is named and fixed for each of the four nodes
- [ ] #2 Each node finishes well inside the timeout in a `-n 4` run, with the time and load recorded
- [ ] #3 The two files that are on no PR lane are laned, or the reason is recorded
<!-- AC:END -->
