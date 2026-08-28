---
id: TASK-574
title: 'Console /rewind: restore-to-before-first-message does not survive restart'
status: In Progress
assignee: []
created_date: '2026-07-25'
updated_date: '2026-08-28 22:08'
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
- [ ] #1 A /rewind restore to before the first message survives an app restart: the conversation resumes showing an empty active path, not the most-recent leaf; all existing turns remain stored and become navigable through the existing branch controls after a new root prompt is sent
- [ ] #2 Conversations with a genuinely-unset pointer (legacy, or never rewound) keep the existing most-recent-leaf resume fallback
- [ ] #3 The persisted representation stays local-only (no sync_log row, matching active_leaf_message_id's write-through)
- [ ] #4 After restart, the deliberately-before-first state restores the selected first prompt's original text into the composer; later unsent edits remain session-only and another restart restores the original text again
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Record the approved local tri-state cursor contract in ADR-100 and the reviewed design spec.
2. Rebase on the latest `dev`, confirm the current schema version, then add the next local-only `active_leaf_before_message_id` migration and atomic cursor persistence API.
3. Hydrate explicit-before-first cursor state and original prompt text through the session draft setter, while preserving unset, persisted-root validation, and invalid-state fallback behavior.
4. Route first-prompt `/rewind` through the dedicated before-message operation, and clear the marker atomically in every durable leaf-advance path, including direct message-acceptance SQL.
5. Update the stale rewind integration fixture's durable Library-policy hydration, then add focused migration, store, UI, integration, and sync-log regression coverage using TDD.
6. Run focused verification, self-review, and document implementation notes.

ADR required: yes

ADR path: `backlog/decisions/100-console-active-path-before-first-cursor.md`

Reason: TASK-574 changes the durable conversation schema, local data ownership, and resume contract.
<!-- SECTION:PLAN:END -->
