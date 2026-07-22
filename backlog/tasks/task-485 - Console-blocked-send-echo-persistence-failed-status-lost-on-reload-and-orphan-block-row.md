---
id: TASK-485
title: >-
  Console blocked-send echo persistence: failed status lost on reload +
  orphan block row
status: To Do
assignee: []
created_date: '2026-07-22 12:19'
labels:
  - console
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up from the task-457(a) code review (finding 3). The cold-send optimistic
echo appends the USER row with `persist=self.store.persistence is not None`
BEFORE the readiness probe, so a blocked first send now creates a durable
persisted conversation (auto-titled from the first attempt) containing a
`failed` USER row — while the honest SYSTEM block-row explaining it is NOT
persisted. Two problems follow:

1. **Status lost on reload (correctness).** When a persisted conversation is
   resumed, `_resume_console_workspace_conversation` reconstructs each message
   with a HARDCODED `status="complete"` (`chat_screen.py:3969`). A row that was
   `failed` (send-blocked) therefore comes back as `complete`, so
   `_provider_messages_for_session(skip_failed=True)` no longer excludes it — the
   never-sent message re-enters the next send's provider context after a
   restart/resume. This defeats the context-exclusion guarantee that
   `mark_message_send_blocked` + `skip_failed` enforce in-session.
2. **Orphan block row (UX).** The `failed` USER row persists but the SYSTEM
   block-row does not, so on reload the user sees a lonely clean-rendered
   "hello" (the `[failed]` suffix is suppressed for USER rows) with no
   explanation; repeated blocked retries accumulate persisted duplicate USER
   rows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A send that never reached the provider does not leave a durable record that, on resume, re-enters the next send's provider context (the never-sent row is either not persisted, or its non-sent state round-trips and is still excluded)
- [ ] #2 Resuming a conversation whose first send was blocked does not show an unexplained lonely user message (no orphan)
- [ ] #3 A successful first send still persists correctly and appears in the workspace rail with its derived title (no regression to task-457(a))
- [ ] #4 Regression coverage for the resume/round-trip path of a blocked send
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Candidate approaches (pick one, needs a UX/design call):

- **Defer echo persistence to the accepted path (recommended).** Append the
  optimistic echo with `persist=False` (in-memory echo still gives the immediate
  cold-send feedback), and persist the USER row + session only once the turn is
  confirmed to proceed (the `_notify_submission_accepted` / assistant-append
  point). A blocked send then persists nothing → no orphan, no reload re-leak.
  Verify the workspace rail still shows a SUCCESSFUL send (rail-refresh depends
  on the session being persisted on accept) and that auto-title (already moved
  before the append in 457(a)) still lands the derived title on the persisted
  conversation.
- **Persist the SYSTEM block-row too + round-trip the failed status.** Keep
  persisting the echo, additionally persist the block-row, and stop hardcoding
  `status="complete"` on reload (`chat_screen.py:3969`) — requires the durable
  message store to carry a per-message failed/blocked flag (schema check).

The first avoids a schema change and removes both problems at once; prefer it
unless there's a product reason to keep blocked attempts in durable history.
<!-- SECTION:PLAN:END -->
