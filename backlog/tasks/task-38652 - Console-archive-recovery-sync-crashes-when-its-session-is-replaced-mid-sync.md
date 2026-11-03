---
id: TASK-38652
title: Console archive-recovery sync crashes when its session is replaced mid-sync
status: To Do
created_date: 2026-11-03 06:50
labels:
- follow-up
- console
- bug
priority: high
references:
- tldw_chatbook/UI/Console_Modules/archive_recovery.py
- tldw_chatbook/Chat/console_chat_store.py
- Tests/UI/test_console_first_chat_first_mount.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while investigating TASK-38651 (2026-11-03, dev 18482c1cf7).

`Tests/UI/test_console_first_chat_first_mount.py::test_start_chatting_through_consoles_first_mount_warns_nothing` failed 1 of 5 runs with `-n 4` at load about 37. It failed at `run_test` exit with `WorkerFailed: KeyError('Unknown Console chat session: <id>')`, raised by `store.messages_for_session(session_id)` (console_chat_store.py:11865) from the archive-recovery view's `sync` (archive_recovery.py:119).

**Likely cause.** `sync` reads `store.active_session_id` once. It then awaits `surface.remove_children()`, `surface.mount(...)`, card removals and `_mount_results`, and only after that calls `store.messages_for_session(session_id)` and `runtime.coordinator.snapshot(session_id)`. If that session was dropped during those awaits, the store raises. The first-chat handoff appears to replace Console's untouched first chat at exactly that point. `sync` runs inside workers that use Textual's default `exit_on_error`, so in the real app this exception would end the app, not just a test.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The failure is reproduced deterministically: a test drops or replaces the active session during one of `sync`'s awaits and shows the KeyError (red) before the fix
- [ ] #2 A `sync` whose session is gone or no longer active after an await stops without error, and the view then shows the current session's state on the next sync
- [ ] #3 Every other read in `sync` and its helpers that uses a captured session id after an await is checked, and fixed the same way where it can raise
- [ ] #4 The first-mount test file passes 10 runs with `-n 4`
<!-- AC:END -->
