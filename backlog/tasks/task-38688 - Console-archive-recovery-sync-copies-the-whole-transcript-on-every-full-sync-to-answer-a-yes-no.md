---
id: TASK-38688
title: "Console archive-recovery sync copies the whole transcript on every full sync to answer a yes/no"
status: Done
assignee:
  - '@claude'
created_date: 2026-11-03 11:40
updated_date: 2026-11-04 01:10
labels:
- follow-up
- performance
- console
priority: low
references:
- tldw_chatbook/UI/Console_Modules/archive_recovery.py
- tldw_chatbook/Chat/console_chat_store.py
- tldw_chatbook/UI/Screens/chat_screen.py
dependencies:
- TASK-24300
- TASK-38652
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the round-5 wave of the 2026-10-23 burn-down (TASK-38652's work, PR #3366, 2026-11-03). Checked on dev 9bbd5c2515.

**Where it runs.** `ConsoleArchiveRecoveryView.sync` (archive_recovery.py:119) runs:
- on every full Console sync once the UI is ready (chat_screen.py:17880);
- as the view's `prepare_view` hook (wiring.py:3717).

**The copy.** When the active chat has no live recovery presentations, sync computes `imported = any("result_set_id" in message.content and "result_ref" in message.content for message in store.messages_for_session(session_id))` (archive_recovery.py:163-166). `messages_for_session` materializes every stream buffer and returns a snapshot of every message (console_chat_store.py:11865-11875). So a sync of a long chat copies the whole transcript to answer one yes/no. Ordinary chats hold no imported recovery text, so `any()` usually walks to the end.

**Prior art.** This is the shape TASK-24300 removed from the keystroke path; that task's guard covers the keystroke path only. The store already has lazy reads (`has_messages` at :11882, `iter_messages_newest_first` at :11929).

PR #3366 adds a `current()` check right before this store read, so a fix should land after it. The cost is not measured; it grows with transcript length.

Depends on TASK-38652 (in PR #3366).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A full Console sync of a chat with no recovery presentations does not snapshot the transcript, pinned by call count as TASK-24300 pinned its sites, and the guard fails when the snapshot read returns
- [x] #2 The imported-recovery copy still shows exactly when some message carries both result_set_id and result_ref, and sync still reads only the chat it captured (TASK-38652's behaviour)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a reader that yields one chat's visible message text without folding, writing or snapshotting, beside the stream-fold helpers in `Chat/console_stream_persistence.py` (the store module is at its size ratchet).
2. `ConsoleArchiveRecoveryView.sync` answers "imported recovery text?" through it instead of `messages_for_session`, still after TASK-38652's `current()` check.
3. Pin in `Tests/UI/test_console_archive_recovery_cards.py` (a census file): `messages_for_session` and `_snapshot` calls during a sync with no presentations are 0 (red on base), and the imported copy shows exactly when one message carries both keys (also in unfolded stream text), not when they are split over two messages.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Approach.** `ConsoleArchiveRecoveryView.sync` now answers "does any message carry both `result_set_id` and `result_ref`?" through `visible_message_texts(store, session_id)`, a new generator in `Chat/console_stream_persistence.py` (beside the stream-fold helpers). It yields each active message's visible text lazily: no fold, no deferred write, no per-message snapshot. The text is what `messages_for_session` would show: unfolded stream text is the joined buffer (the TASK-259 invariant, as `read_only_messages_for_session` reads it), and a quarantined message is skipped because its snapshot shows only the placeholder. The read still runs after TASK-38652's `current()` check and asks only for the chat the sync captured, synchronously inside `any()`, so no await can switch the chat under it.

**Decision.** A helper module rather than a store method: `Chat/console_chat_store.py` sits exactly at its module-size ratchet row (22,445), and the ratchet's own guidance is to put new code elsewhere.

**Test fakes.** Three cards tests faked the store as a `SimpleNamespace` with `messages_for_session`; the view now reads the store's own transcript, so they use a real empty store (`_empty_source_store`).

**Evidence.** `Tests/UI/test_console_archive_recovery_cards.py::test_sync_answers_imported_recovery_without_copying_the_transcript` (none/split/saved/streamed) counts `messages_for_session` and `ConsoleChatStore._snapshot` calls only around two syncs (the setup's appends are not billed). Red on base c2cdb6b82e in all 4 cases: "two syncs of a 40-message chat made 2 messages_for_session calls and 80 message snapshots" (41/42 messages: 82/84). Green after: 0 copies, and the imported copy shows for saved and unfolded stream text, not for none or for the two keys split over two messages (AC#2). TASK-38652's `test_sync_stops_when_its_chat_is_replaced_mid_sync[controls|imported]` and the flow file's `test_native_result_refresh_remains_manual_and_imported_json_is_inert` stay green. Cards file: 32 passed (`-n 4`).

**Files.** `tldw_chatbook/Chat/console_stream_persistence.py`, `tldw_chatbook/UI/Console_Modules/archive_recovery.py`, `Tests/UI/test_console_archive_recovery_cards.py`.
<!-- SECTION:NOTES:END -->
