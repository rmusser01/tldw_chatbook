---
id: TASK-32467
title: >-
  Console span note actions bypass the ephemeral registry (summarize-note /
  save-transcript-note)
status: To Do
assignee: []
created_date: '2026-09-11 23:26'
labels:
  - console
  - notes
  - ephemeral
  - critique-notes-2026-09
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-31759's two More-menu note actions, Summarize up to here as note (summarize-note) and Save transcript up to here as note (save-transcript-note), write a Local Note but have no row in EPHEMERAL_BLOCKED_ACTIONS (Chat/console_ephemeral.py), and ConsoleMessageActionService._action_enabled never calls blocked_reason for them -- so a temporary chat, which promises nothing is written locally, writes a note through either action today. Found by task-32146, whose own capture-note carries both the registry row and a dispatch-time re-check in UI/Console_Modules/message.py. The registry tests in Tests/Chat/test_console_ephemeral.py iterate the registry's own keys, so they cannot catch a missing row.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both actions are offered disabled with the registry's reason while the Console chat is temporary
- [ ] #2 Each action's dispatch branch refuses with the same reason and starts no note worker, pinned RED to GREEN on the real handle_console_message_action route
- [ ] #3 No note is written from a temporary chat by either action
<!-- AC:END -->
