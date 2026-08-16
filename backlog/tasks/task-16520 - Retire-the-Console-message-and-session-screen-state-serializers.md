---
id: TASK-16520
title: Retire the Console message/session screen-state serializers
status: To Do
assignee: []
labels:
  - console
  - cleanup
priority: medium
---

## Description

task-15860 Task 3 made the app-owned `ConsoleRuntime` store the single
source of truth for Console message history: `ChatScreen`'s
`ScreenStateStore` snapshot no longer carries `sessions`,
`messages_by_session` or `active_session_id`, and
`_restore_native_console_state` no longer calls
`ConsoleChatStore.restore_state`.

That left a cluster of (de)serializers with **no production caller**. They
are still exercised by ~45 tests, so the coverage they carry now describes
a code path the app never takes — which is worse than no coverage, because
it reads as protection. Retiring them is mechanical but wide, and was
deliberately kept out of the continuity landing so that the riskiest change
in the arc stayed small and revertible.

The cluster, all currently unreached from production:

- `ChatScreen._serialize_console_message` / `_restore_console_message`
  (delegators) and `ConsoleMessageController`'s implementations
- `ChatScreen._rehydrate_console_message_image` /
  `_rehydrate_console_message_attachments` /
  `_rehydrate_console_message_generation_metadata` (delegators) and the
  controller implementations, where they are not also used by the DB-resume
  path (`restore_persisted_session` hydrates generation metadata itself —
  check each before deleting)
- `ConsoleSessionController._console_session_to_state` /
  `_console_session_from_state`
- `ChatScreen._CONSOLE_PENDING_STASH_ATTR`'s H3 filter
  (`_filter_h3_attachment_from_app_stash`) IF the stash itself is retired —
  the store's own `consume_pending_attachment` already removes a completed
  H3 edit's pending, so the stash's only remaining job may be redundant

## Acceptance Criteria

- [ ] No production code path constructs or consumes a serialized Console
      message or session dict
- [ ] Behaviour asserted by the retired tests is either still covered
      against the surviving-store mechanism, or explicitly recorded as a
      behaviour that no longer exists (per test, not in bulk)
- [ ] Console message continuity across a navigation, a same-target
      navigation and a rapid route-switch soak stays green
      (`Tests/UI/test_console_store_continuity.py`)
- [ ] Pending attachments still survive a navigation exactly once (no
      duplication, no loss), including the completed-H3-edit case
