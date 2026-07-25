---
id: TASK-549
title: 'Console /rewind: guard the modal callback against a changed active session'
status: Done
assignee: []
created_date: '2026-07-24'
labels:
  - console
  - tech-debt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`ChatScreen._apply_console_rewind_choice` (PR #844) captures `session_id` when the rewind modal opens. If the active session could change while the modal is up, restore would mutate the captured session's active leaf while the composer refill and `_sync_native_console_chat_ui` operate on the then-current session. In practice a ModalScreen blocks session switching, so this is theoretical today — but a one-line `if store.active_session_id != session_id: notify + return` guard at the top of the callback makes the flow robust against future modal/timing changes (e.g. a background auto-switch or a future non-modal rewind surface). Same guard applies to the summarize branch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The rewind choice callback no-ops with a notify when the active session differs from the one the modal was opened for
- [x] #2 Covered by a unit test (fake a session switch between open and dismiss)
<!-- AC:END -->

## Implementation Notes

Added an active-session guard at the top of `_apply_console_rewind_choice`: on a mismatch between the store's active session and the session captured at modal-open, the callback notifies ('Console session changed — rewind cancelled.') and returns with zero mutation — covering restore, summarize, and None branches. Unit tests fake a session switch between open and dismiss for both actions.
