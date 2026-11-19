---
id: TASK-42017
title: "The second-Enter-after-unknown-command test is flaky under load"
status: To Do
assignee: []
created_date: 2026-11-19 19:39
updated_date: 2026-11-19 19:39
labels:
- follow-up
- tests
- console
priority: low
references:
- Tests/UI/test_console_send_admission_off_pump.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41904 (round 17). `test_a_second_enter_after_an_unknown_command_hint_sends_it_as_text` failed about 2 of 4 serial runs at base, at load 42-55.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The cause is named and the test passes 10 of 10 at `-n 4` with the load recorded
<!-- AC:END -->
