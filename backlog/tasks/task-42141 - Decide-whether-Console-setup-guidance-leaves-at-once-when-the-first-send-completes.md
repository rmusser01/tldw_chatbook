---
id: TASK-42141
title: "Decide whether Console setup guidance leaves at once when the first send completes"
status: To Do
assignee: []
created_date: 2026-11-20 20:06
updated_date: 2026-11-20 20:06
labels:
- follow-up
- console
- onboarding
- decision
priority: low
references:
- tldw_chatbook/UI/Screens/chat_screen.py
- tldw_chatbook/Chat/console_received_dispatch.py
- tldw_chatbook/Chat/console_onboarding_state.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the consoletests stream (TASK-42015, round 18). `ChatScreen._console_first_send_completed` (`tldw_chatbook/UI/Screens/chat_screen.py:11840-11857`) reads `console.onboarding.first_send_completed` from `app.app_config` once and keeps it in `_console_first_send_completed_cached` for the screen's life. The live save after a first reply goes through `ConsoleRuntime._record_successful_first_send` (it writes the flag with `apply_settings_mutation_to_cli_config`, `Chat/console_received_dispatch.py:952`) and does not touch that memo. The memo's only other writer, `ChatScreen._record_console_first_send` (`:11859`), has no production callers (TASK-42015 deletes it).

So on the same ChatScreen, a new empty tab opened after the first reply still shows the setup guidance (`first_send_completed=self._console_first_send_completed()` at `:13856`); a fresh ChatScreen reads the saved flag and shows none. Product question: should guidance leave as soon as the first send completes?
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The owner's decision is recorded: whether setup guidance leaves at once when the first send completes, or stays for the rest of that ChatScreen's life as today
- [ ] #2 If at once: a new empty tab opened after the first reply on the same ChatScreen shows no setup guidance, and a test shows it, red before the change
- [ ] #3 If today's behaviour stays: a comment at the memo records the decision, and a test pins that the guidance stays on that ChatScreen
<!-- AC:END -->
