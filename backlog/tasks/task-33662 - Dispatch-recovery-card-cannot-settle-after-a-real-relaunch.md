---
id: TASK-33662
title: Dispatch-recovery card cannot settle after a real relaunch
status: To Do
assignee: []
created_date: '2026-10-01 21:10'
labels:
  - console
  - recovery
  - bug
dependencies:
  - TASK-33661
references:
  - tldw_chatbook/Chat/console_conversation_hydration.py
  - tldw_chatbook/Chat/console_chat_controller.py
  - Tests/Chat/test_console_dispatch_recovery.py
  - qa/task-33661-resend/README.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Why: found while building TASK-33661 (Resend) on 2026-10-01; this bug predates that change.

If the app quits while a reply is still in flight, the next launch shows a dispatch-recovery card ("Retry response" / "Retry anyway" / "Discard"). After a REAL relaunch, both Retry and Discard refuse with "That response recovery action is unavailable." That leaves the chat stuck: sends stay refused with "Finish or discard the pending response…", which is exactly the broken-chat case the owner wants to be able to resume.

Cause, as traced by the implementer:
- The recovery state keeps the persisted assistant message id.
- Hydration (console_messages_from_conversation_tree → _ingest_full_tree) gives every restored node a fresh native id.
- claim_dispatch_recovery_action → _message_or_raise then raises KeyError.

Reproduced in a controller test and live; see the e2/e3 captures in qa/task-33661-resend/.

Tests/Chat/test_console_dispatch_recovery.py restores nodes whose native id equals the persisted id, so the suite never sees the bug.

Until this is fixed, the Resend path "Discard, then Resend" works only within a single app session.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After a real relaunch, Retry response and Discard on a dispatch-recovery card both settle the recovery. Retry streams into the pending reply; Discard settles it as discarded and keeps the user message.
- [ ] #2 After Discard following a relaunch, the user message offers Resend (TASK-33661), and Resend re-runs the turn in place.
- [ ] #3 A regression test drives the production hydration path, where restored nodes get fresh native ids. It fails on the current code and passes after the fix.
- [ ] #4 Every existing dispatch-recovery test still passes, and any test that relied on native id == persisted id is rewritten on purpose and named in the notes.
<!-- AC:END -->
