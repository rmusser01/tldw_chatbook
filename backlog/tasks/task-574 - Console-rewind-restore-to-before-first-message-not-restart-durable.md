---
id: TASK-574
title: 'Console /rewind: restore-to-before-first-message does not survive restart'
status: Done
assignee: []
created_date: '2026-07-25'
updated_date: '2026-08-29 01:47'
labels:
  - console
  - chat
  - rewind
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Documented v1 limitation of the `/rewind` menu (SP2, PR #844): restoring to before the FIRST message clears the active leaf via `set_active_leaf(None)`, but the persisted `active_leaf_message_id` column treats NULL as "unset", so resume falls back to the most-recent leaf — the restore silently un-does itself across an app restart. Within the running session the behavior is correct. Closing this needs a way to persist "deliberately before the first message" distinctly from "no pointer stored" (a sentinel value or an additional local-only column; either way the pointer stays local-only and must not sync, matching the Phase A design).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A /rewind restore to before the first message survives an app restart: the conversation resumes showing an empty active path, not the most-recent leaf, and no existing message is deleted or rewritten
- [x] #2 Conversations with a genuinely-unset pointer (legacy, or never rewound) keep the existing most-recent-leaf resume fallback
- [x] #3 The persisted representation stays local-only (no sync_log row, matching active_leaf_message_id's write-through)
- [x] #4 After restart, the deliberately-before-first state restores the selected first prompt row's current durable text into the composer; later unsent edits remain session-only and another restart restores that durable text again
- [x] #5 If a persisted conversation cannot save the before-first cursor, the running session keeps the empty active path and composer refill while warning that the restart position was not saved
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Record the approved local tri-state cursor contract in ADR-100 and the reviewed design spec.
2. Rebase on the latest `dev`, confirm the current schema version, then add the next local-only `active_leaf_before_message_id` migration and atomic cursor persistence API.
3. Hydrate explicit-before-first cursor state and the target row's current durable prompt text through the session draft setter, while preserving unset behavior, pre-repair imported-root validation, empty-tree cleanup, and invalid-state fallback.
4. Route first-prompt `/rewind` through the dedicated before-message operation, warn without rolling back an in-memory rewind when its durable write fails or its persisted target lacks a durable ID, and clear the marker atomically in every durable leaf-advance path, including direct message-acceptance SQL.
5. Update the stale rewind integration fixture's durable Library-policy hydration, then add focused migration, store, UI, integration, and sync-log regression coverage using TDD.
6. Verify canonical root-branch recovery and legacy flat-tree non-deletion separately, then run focused verification, self-review, and document implementation notes.

ADR required: yes

ADR path: `backlog/decisions/100-console-active-path-before-first-cursor.md`

Reason: TASK-574 changes the durable conversation schema, local data ownership, and resume contract.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the v54 migration and atomic two-column cursor API in tldw_chatbook/DB/ChaChaNotes_DB.py; threaded the local-only cursor through tldw_chatbook/Chat/console_chat_store.py and console_conversation_hydration.py; routed the first-prompt UI through the honest boolean result in tldw_chatbook/UI/Screens/chat_screen.py; and cleared the marker inside the guarded acceptance transaction in console_dispatch_repository.py. The local message-ID reference deliberately avoids persisting draft text or expanding Sync, Chatbook, fork, or trajectory formats; ambiguous legacy flat trees guarantee durable non-deletion rather than a repaired branch shape. Focused evidence before completion: 245 passed; scripts/preflight.sh passed; git diff --check origin/dev...HEAD and git diff --check passed. Coverage includes canonical restart/resend, unset fallback, invalid repair, persistence failures, temporary sessions, session-only draft edits, attachment-only text, and legacy non-deletion. ADR: backlog/decisions/098-console-active-path-before-first-cursor.md.
<!-- SECTION:NOTES:END -->
