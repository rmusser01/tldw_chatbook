---
id: TASK-16810
title: Ctrl+K session switcher crashes the app (AttributeError on chat_screen)
status: Done
assignee:
  - '@claude'
created_date: '2026-08-15'
updated_date: '2026-08-16 16:24'
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
- [x] #1 Pressing Ctrl+K with the Console tab active opens the session switcher modal instead of exiting the app
- [x] #2 The switcher's persisted-rows sync receives the actual current conversation id (the same value `self._session._current_console_conversation_id()` returns), so the active session is marked/positioned correctly in the list
- [x] #3 A regression test drives the Ctrl+K action (or calls `action_open_console_session_switcher` directly) on a mounted chat screen and asserts the app stays alive and the modal is pushed
- [x] #4 A quick audit of `action_open_console_session_switcher`'s sibling actions confirms no other `self._current_console_*` call bypasses the `_session`/`_workspace` delegation the rest of the file uses (fix any found or file them)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Red: run the two pre-existing chat-flow failures (test_console_native_chat_flow.py) that hit this crash; add a direct regression test driving action_open_console_session_switcher
2. Fix: delegate through self._session._current_console_conversation_id() at chat_screen.py:2986 (matches the file's own working call at :9965)
3. Audit sibling actions for other missed _session/_workspace delegations (AC#4)
4. Green: targeted suites; PR
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Already fixed on dev before this task was picked up: commit 386c66a6f (task-16792, another session, 2026-08-16) restored the ChatScreen delegation seam _current_console_conversation_id -> self._session, which both the Ctrl+K switcher (broken by 520b1ec12) and /research dispatch (broken by e1f3a4424) call. Verified at dev tip c2f30862c: the formerly dev-red switcher test pair in Tests/UI/test_console_native_chat_flow.py (2 tests, -k switcher) passes; those tests drive the action and assert the modal opens with the active conversation id, covering AC1-AC3. AC4 audit: the only two bare self._current_console_conversation_id() call sites (switcher action, /research delivery) now resolve through the restored seam, and the _current_console_rail_* variants are defined on the screen itself - no other missed delegation of this family found. No code change needed; closing as duplicate of the landed task-16792 fix (its docstring also cites task-16815 for the same defect).
<!-- SECTION:NOTES:END -->
