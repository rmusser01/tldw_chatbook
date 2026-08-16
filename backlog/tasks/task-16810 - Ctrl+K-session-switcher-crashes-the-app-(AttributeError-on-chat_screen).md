---
id: TASK-16810
title: Ctrl+K session switcher crashes the app (AttributeError on chat_screen)
status: To Do
assignee: []
created_date: '2026-08-15'
labels:
  - console
  - bug
  - crash
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Pressing **Ctrl+K** in Console exits the whole application. `action_open_console_session_switcher`
(`tldw_chatbook/UI/Screens/chat_screen.py:2986`) calls
`self._current_console_conversation_id()`, but no such method exists on the
screen — the real method lives on the session module
(`UI/Console_Modules/session.py:1960`; the screen's own working call at
`chat_screen.py:9965` correctly goes through `self._session.…`). The
resulting `AttributeError` escapes a Textual action handler, which routes to
`app._handle_exception()` and terminates the app.

Discovered live on 2026-08-15 while verifying the turn-file-card branch; the
defect is present at that branch's merge base (`5911e5f4c`), i.e. it is a
pre-existing dev bug, not introduced by the card work. Likely a missed
delegation during the chat_screen decomposition (methods moved onto
`_session`/`_workspace` modules).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Pressing Ctrl+K with the Console tab active opens the session switcher modal instead of exiting the app
- [ ] #2 The switcher's persisted-rows sync receives the actual current conversation id (the same value `self._session._current_console_conversation_id()` returns), so the active session is marked/positioned correctly in the list
- [ ] #3 A regression test drives the Ctrl+K action (or calls `action_open_console_session_switcher` directly) on a mounted chat screen and asserts the app stays alive and the modal is pushed
- [ ] #4 A quick audit of `action_open_console_session_switcher`'s sibling actions confirms no other `self._current_console_*` call bypasses the `_session`/`_workspace` delegation the rest of the file uses (fix any found or file them)
<!-- AC:END -->
