---
id: TASK-38688
title: "Console archive-recovery sync copies the whole transcript on every full sync to answer a yes/no"
status: To Do
created_date: 2026-11-03 11:40
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
- [ ] #1 A full Console sync of a chat with no recovery presentations does not snapshot the transcript, pinned by call count as TASK-24300 pinned its sites, and the guard fails when the snapshot read returns
- [ ] #2 The imported-recovery copy still shows exactly when some message carries both result_set_id and result_ref, and sync still reads only the chat it captured (TASK-38652's behaviour)
<!-- AC:END -->
