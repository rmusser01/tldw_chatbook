---
id: TASK-574
title: 'Console /rewind: restore-to-before-first-message does not survive restart'
status: To Do
assignee: []
created_date: '2026-07-25'
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
- [ ] #1 A /rewind restore to before the first message survives an app restart: the conversation resumes showing an empty active path (with all turns recoverable by swipe/rewind), not the most-recent leaf
- [ ] #2 Conversations with a genuinely-unset pointer (legacy, or never rewound) keep the existing most-recent-leaf resume fallback
- [ ] #3 The persisted representation stays local-only (no sync_log row, matching active_leaf_message_id's write-through)
<!-- AC:END -->
