---
id: TASK-38751
title: "Console a hook check refused because the chat owner changed tells the user to review or disable hooks"
status: To Do
assignee: []
created_date: 2026-11-04 06:34
labels:
- follow-up
- console
- ux
- hooks
priority: medium
references:
- tldw_chatbook/Chat/console_automatic_submission.py
- tldw_chatbook/Chat/console_chat_controller.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 7 of the 2026-10-23 follow-up burn-down. Checked on dev bb80905eb3, 2026-11-04.

`read_hook_admission_result` (`Chat/console_automatic_submission.py:449`) runs the hook preparation read for a send. After the read, it checks that the hook admission owner has not changed: the chat session and its binding revision, the runtime, and its hook permissions. If any of them moved, it raises `RuntimeError("Console hook admission owner changed.")` (:546).

The function's catch-all (:559-562) turns every exception into "Hooks unavailable; review or disable hooks before sending." So a send refused because the user switched chats during the read, or because the Console runtime was disposed, tells them to review or disable their hooks. Nothing is wrong with the hooks, and disabling them is the wrong advice. The same copy is also the generic fallback of `hook_admission_reason` (:400).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A send refused because the hook admission owner changed shows copy that says what happened (the chat changed during the hook check) and what to do (send again), with no advice about hooks
- [ ] #2 A real failure of the hooks read still shows the review-or-disable copy
- [ ] #3 A test changes the owner during the preparation read and checks the copy; it is red before the change
- [ ] #4 Any User Guide page that describes hook send refusals matches the new copy
<!-- AC:END -->
