---
id: TASK-23144
title: >-
  Bare ChatScreen shells miss the Library-activity controller and 46 tests die
  in setup
status: To Do
assignee: []
created_date: '2026-08-28'
labels:
  - tests
  - console
priority: high
dependencies: []
---

## Description

46 tests fail with `AttributeError: 'ChatScreen' object has no attribute '_library_activity'` —
every test in `Tests/UI/test_console_citation_sources.py` (41) and
`Tests/UI/test_console_composer_menu.py` (5). Production is correct: the controller is installed by
`build_console_controllers`, which only runs from `ChatScreen.__init__`, and these tests build
shells via `ChatScreen.__new__(ChatScreen)`. They die during **setup**, before asserting anything.

The sharp part: this is the exact failure mode `Tests/Architecture/test_bare_chat_screen_shells_wire_the_fleet.py`
was written to ratchet (TASK-21381, which fixed 115 such failures across 8 files). The guard is an
AST scan hard-coded to look for `stub_fleet_controller` only, so it cannot see a **second**
controller entering the same kwargs build. Widening the guard is the durable half of this task —
without it the next controller added to that build repeats this.

## Acceptance Criteria

- [ ] Both files pass, with bare shells wiring the Library-activity controller through a shared stub
- [ ] The architecture ratchet asserts that **every** controller the chat-store wiring reads is
  stubbed — not just the fleet controller — so a newly added controller fails at the guard rather
  than in dozens of unrelated tests
- [ ] The widened guard is proven by a negative control (removing a stub makes it fail)

## Evidence

Installed at `tldw_chatbook/UI/Console_Modules/wiring.py:751`. The setup chain that dies:
`screen._console_chat_store = ...` -> `_console_runtime().set_chat_store()` ->
`_ensure_console_chat_controller` (`chat_screen.py:5008`) -> `chat_screen.py:5116`
`"_library_provider_factory": self._library_activity.build_provider`. Shells built at
`Tests/UI/test_console_citation_sources.py:480`. Guard's blind spot: `_calls_fleet_stub`,
`Tests/Architecture/test_bare_chat_screen_shells_wire_the_fleet.py:87`.

Introduced by `d8d5f9f2b1` (2026-08-27) "feat(console): capture and review minimized Library
activity (TASK-19900.5) (#2154)", which updated 3 UI test files but not these two.
