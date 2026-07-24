---
id: TASK-546
title: 'Console /rewind: guard the modal callback against a changed active session'
status: To Do
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
- [ ] #1 The rewind choice callback no-ops with a notify when the active session differs from the one the modal was opened for
- [ ] #2 Covered by a unit test (fake a session switch between open and dismiss)
<!-- AC:END -->
