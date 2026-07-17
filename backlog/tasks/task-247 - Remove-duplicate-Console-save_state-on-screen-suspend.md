---
id: TASK-247
title: Remove duplicate Console save_state on screen suspend
status: Done
assignee: []
created_date: '2026-07-16 14:30'
updated_date: '2026-07-17 00:20'
labels: [performance, console]
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
app.py:4611-4621 calls save_state() and stores the result before switch_screen; Textual's ScreenSuspend then fires ChatScreen.on_screen_suspend (chat_screen.py:10651-10655) which calls save_state() AGAIN and discards the result — a full O(sessions×messages) native-console serialization wasted on every tab switch away from Console. Full analysis with measurements: Docs/Design/2026-07-16-performance-audit.md (§P0 A2).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Console save_state runs exactly once per tab-switch-away (test counts invocations)
- [x] #2 State restoration behavior unchanged (existing round-trip tests green)
<!-- AC:END -->

## Implementation Notes

Removed `ChatScreen.on_screen_suspend` (chat_screen.py, was ~10651-10655)
outright rather than turning it into a no-op. Textual dispatches both a
private `_on_screen_suspend` (defined on `Screen` itself -- toggles the
suspended-screen CSS class, clears mouse-over/tooltip state) and, if
present, a public `on_screen_suspend` override, independently, by walking
the full MRO (`MessagePump._get_dispatch_methods`). Deleting the override
therefore only removes the redundant `save_state()` call; `Screen`'s own
suspend bookkeeping is untouched since it lives under the differently-named
private handler. `app.py` (~4611-4624) already calls `save_state()` once,
explicitly, before switching screens away from Console and stores that
return value -- the deleted override called it again and threw the result
away.

New test `Tests/UI/test_chat_screen_suspend.py`. RED before the fix: a bare
`ChatScreen.__new__(ChatScreen)` instance with `save_state` replaced by a
call-recorder showed `on_screen_suspend()` invoking it (assertion failure),
and `"on_screen_suspend" in ChatScreen.__dict__` was `True`. GREEN after:
both tests pass (2 passed) -- the getattr-based lookup finds nothing to
call, and the override is gone from the class dict entirely. Full regression
check: `Tests/UI/test_chat_screen_state.py` + `Tests/UI/test_console_native_chat_flow.py`
-- 204 passed.
