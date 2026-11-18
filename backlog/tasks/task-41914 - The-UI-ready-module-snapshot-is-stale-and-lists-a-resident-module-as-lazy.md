---
id: TASK-41914
title: "The UI-ready module snapshot is stale and lists a resident module as lazy"
status: To Do
assignee: []
created_date: 2026-11-18 20:31
updated_date: 2026-11-18 20:31
labels:
- follow-up
- tests
- performance
priority: low
references:
- Tests/Performance/test_ui_ready_module_census.py
- Tests/UI/test_console_first_send_preimport.py
dependencies:
- TASK-36231
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41512 (round 16): the UI-ready census reads drift +3/-3 against its snapshot. TASK-41308 made `MCP.console_snapshot` resident at ready, but `LIVE_FIRST_SEND_IMPORTS` (`Tests/UI/test_console_first_send_preimport.py` ~42-47) still lists it as lazy, so refreshing the snapshot fails `test_the_send_guard_counts_a_module_resident_at_ui_ready_as_warm`.
 TASK-36231 owns re-pinning the snapshot and the membership-drift check; this task is only the `LIVE_FIRST_SEND_IMPORTS` entry. The production warm-up list `FIRST_SEND_MODULES` is out of scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `MCP.console_snapshot` leaves `LIVE_FIRST_SEND_IMPORTS`, so a snapshot refresh under TASK-36231 no longer fails the send-guard test
- [ ] #2 The send-guard test passes
<!-- AC:END -->
