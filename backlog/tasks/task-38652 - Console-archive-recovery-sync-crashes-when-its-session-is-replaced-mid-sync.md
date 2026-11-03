---
id: TASK-38652
title: Console archive-recovery sync crashes when its session is replaced mid-sync
status: Done
assignee:
  - '@claude'
created_date: 2026-11-03 06:50
updated_date: 2026-11-03 07:04
labels:
- follow-up
- console
- bug
priority: high
references:
- tldw_chatbook/UI/Console_Modules/archive_recovery.py
- tldw_chatbook/Chat/console_chat_store.py
- Tests/UI/test_console_first_chat_first_mount.py
- Tests/UI/test_console_archive_recovery_cards.py
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
- [x] #1 The failure is reproduced deterministically: a test drops or replaces the active session during one of `sync`'s awaits and shows the KeyError (red) before the fix
- [x] #2 A `sync` whose session is gone or no longer active after an await stops without error, and the view then shows the current session's state on the next sync
- [x] #3 Every other read in `sync` and its helpers that uses a captured session id after an await is checked, and fixed the same way where it can raise
- [x] #4 The first-mount test file passes 10 runs with `-n 4`
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Red first, in the census-listed `Tests/UI/test_console_archive_recovery_cards.py`: mount the view over a real `ConsoleChatStore` and replace the active chat (switch to a second one, close the first, as the first-chat handoff does) inside one of `sync`'s surface mounts. Case 1, during the controls mount: `sync` raises the KeyError. Case 2, during the imported-recovery mount, with the new chat's confirmation presented in that window: the old sync clears it.
2. Fix in `sync`: after its awaits, re-check that the captured session is still the store's active one before the store read (`messages_for_session`) and the coordinator reads that write the controls; otherwise return. The next sync sees the session change and remounts for the current chat.
3. Audit the other post-await reads in `sync` and its helpers (`_mount_results`, `_mount_receipt`, `show_confirmation`, `show_receipt`, `decide`, `action`, `receipt_action`, `_revise_workspace_name`), and fix only the ones that can raise or render another chat's state.
4. Green: the new tests and the rest of the archive-recovery card and flow files. Then the first-mount file 10 times with `-n 4`, the size ratchets, and preflight.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Cause, confirmed.** `sync` captures `store.active_session_id`, then awaits surface mounts. The first-chat handoff runs on the same loop: it calls `store.switch_session` to its own chat, then closes the untouched one (`rollback_created_pristine_session`). When `sync` resumed, it read the closed chat with `store.messages_for_session` and raised `KeyError('Unknown Console chat session: ...')`.

**Fix** (`UI/Console_Modules/archive_recovery.py`). A local `current()` check in `sync`: the surface is attached and the captured chat is still the store's active one.
- It replaces the existing attachment check before the store read. It is also checked after the imported-recovery mount, before the coordinator reads that write the controls.
- On a mismatch, `sync` returns. `self._session_id` then differs from the active chat, so the next sync remounts the current chat.
- "Still active" also means "still known" here. Every store path that removes a chat moves `active_session_id` off it in the same synchronous call (console_chat_store.py: `_close_session_locked` (:5251 then `_activate_session` at :5144), `rollback_created_pristine_session` (:2952), `rollback_restored_session` (:2977), `register_fork_snapshot`'s exception rollback (:8634-8649) and `restore_state` (`_activate_session(None)` at :10630 before the clear at :10651)). `sync` resumes on the UI loop between such calls. So no separate membership check is needed.

**Audit of other post-await reads (AC#3).**
- `runtime.presentations(session_id)` after the remount awaits is a dict `.get`, so it cannot raise. Its records go to `_mount_results`, which refuses a chat that is not answerable (that is, not active).
- `coordinator.snapshot` and `show_confirmation` after the library-button mount cleared a confirmation just presented for the new chat. The new `imported` case shows this. Fixed by the second check.
- `coordinator.receipts` is a `.get`. `_mount_receipt` refuses a chat that is not answerable. After the receipt awaits, only `surface.display` is written, from this sync's own rows, and the next sync remounts. Left as is.
- Each of the following re-checks generation and answerability after its own await, or ends in a fresh `sync()`, so none needed a change: `_mount_results`, `_mount_receipt`, `show_receipt`, `decide`, `action`, `_revise_workspace_name`, `receipt_action`, `open_library`.

**Test.** `Tests/UI/test_console_archive_recovery_cards.py::test_sync_stops_when_its_chat_is_replaced_mid_sync[controls|imported]`, in a census-listed file. It uses a real `ConsoleChatStore`, and the patched surface mount switches to the handoff chat and closes the untouched one, as the handoff does.
- Red before the fix:
  - `controls`: `KeyError: 'Unknown Console chat session: ...'` at `archive_recovery.py:166`, the failure seen in TASK-38651's base runs.
  - `imported`: `assert None is <pending confirmation>`, the new chat's confirmation cleared by the old sync.
- Green after the fix, with the next sync showing the handoff chat's confirmation and no stale library button.

**Runs** (`-n 4`; the load average was 30-53 throughout):
- `test_console_archive_recovery_cards.py`: 28 passed.
- Size ratchets (`Tests/Architecture/test_screen_size_ratchet.py`, `test_module_size_ratchet.py`, serial): 205 passed.
- ruff format and check are clean.
- AC#4: the first-mount file passed 10 of 10 runs (9 of 9 tests each), at load 30-42.
- Preflight passed.
- `test_console_archive_recovery_flow.py` (in no PR lane) shows the same reds at base source with the same command, so they are not from this change:
  - `test_retained_native_recovery_resource_owners_plateau_and_close` (`assert 1 >= 2`) fails every run.
  - Two to four tests fail per run in the `mounted_recovery` setup with `_ConsoleTurnRefusedError` ("Console submit requires attention ... status=refused"). With the fix, the two from the first run passed 12 of 12 on their own.
- `Tests/Performance/test_ui_ready_module_census.py` (1032 > 1031 modules) is also red at base source. This change adds no imports.
<!-- SECTION:NOTES:END -->
