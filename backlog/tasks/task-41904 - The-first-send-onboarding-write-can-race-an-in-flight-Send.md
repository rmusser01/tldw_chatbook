---
id: TASK-41904
title: "The first-send onboarding write can race an in-flight Send"
status: To Do
assignee: []
created_date: 2026-11-18 20:31
updated_date: 2026-11-18 20:31
labels:
- follow-up
- console
priority: medium
references:
- tldw_chatbook/UI/Screens/chat_screen.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41511's harness (round 16). The first-ever reply writes the first-send onboarding flag to config (`ChatScreen._record_console_first_send` and `_save_console_onboarding_flag`, chat_screen.py ~11859-11888, a thread worker calling `save_setting_to_cli_config`); a Send in flight at that moment (most plausibly a queued prompt admitted right after the first reply) fails with "Draft or chat changed; Send again.". The test harness now marks onboarding done to avoid it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A queued Send admitted during the first-send onboarding write is not refused
- [ ] #2 A test reproduces the race and is red before the fix
<!-- AC:END -->
