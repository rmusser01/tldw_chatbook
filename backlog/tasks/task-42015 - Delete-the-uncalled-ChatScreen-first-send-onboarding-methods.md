---
id: TASK-42015
title: "Delete the uncalled ChatScreen first-send onboarding methods"
status: To Do
assignee: []
created_date: 2026-11-19 19:39
updated_date: 2026-11-19 19:39
labels:
- follow-up
- console
- cleanup
priority: low
references:
- tldw_chatbook/UI/Screens/chat_screen.py
- Tests/UI/test_console_native_chat_flow.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41904 (round 17). `ChatScreen._record_console_first_send` and `_save_console_onboarding_flag` have no production callers; only `test_console_record_first_send_repairs_corrupt_config_value` calls the first. The live save is `_persist_first_send_when_quiescent`, whose `_record_successful_first_send` already guards a non-dict `console` or `onboarding`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both methods and the test that keeps them alive are removed, with the corrupt-value case covered on the live path if it is not already
<!-- AC:END -->
