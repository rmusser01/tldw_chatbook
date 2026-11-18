---
id: TASK-41903
title: "A Console goal iteration from a cold Console captures skill trust synchronously"
status: To Do
assignee: []
created_date: 2026-11-18 20:31
updated_date: 2026-11-18 20:31
labels:
- follow-up
- console
priority: medium
references:
- tldw_chatbook/Chat/console_runtime.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41507 (round 16). `ConsoleRuntime._submit_goal_iteration` captures its configuration synchronously, as the launch wake did, so a goal iteration from a cold Console would hit the `console_capture_source_not_ready` refusal TASK-41507 fixed for wakes. Untested.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A goal iteration from a cold Console reaches the provider or shows a correct refusal, never `console_capture_source_not_ready`
- [ ] #2 A test starts cold and is red before the fix
<!-- AC:END -->
