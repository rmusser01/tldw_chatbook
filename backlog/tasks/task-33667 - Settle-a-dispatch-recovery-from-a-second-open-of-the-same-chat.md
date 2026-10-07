---
id: TASK-33667
title: Settle a dispatch recovery from a second open of the same chat
status: To Do
assignee: []
created_date: '2026-10-02 19:50'
labels:
  - console
  - recovery
dependencies:
  - TASK-33662
references:
  - tldw_chatbook/Chat/console_chat_store.py
  - tldw_chatbook/UI/Console_Modules/workspace.py
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Why: Qodo flagged this on #2971 (TASK-33662). TASK-33662 made a restored recovery owner keep its persisted id as its native id, so Retry and Discard work after a relaunch. Native ids are unique store-wide, though. When the same conversation is opened a second time (`open_console_workspace_conversation_by_id` defaults to `reuse_existing=False`), the second session's owner rows get fresh ids.

The second session's recovery card then refuses with "That response recovery action is unavailable", and its pending recovery keeps blocking sends in that tab. Before TASK-33662 every open refused this way; now only the second does. Closing that tab and using the first works.

A full fix resolves the recovery's persisted owner ids to each session's native ids. It needs a session-local persisted-to-native owner map that every recovery path uses: claim, retry, discard, release, settle, the generation-token bind, the continuation handoff, and the controller's `before_message_id` and `get_message` reads.

An alternative is to stop opening the same conversation twice: reuse the open session, as the `reuse_existing=True` path already does.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With the same chat open twice after a relaunch, Retry and Discard settle the recovery from either open, without changing the other session's rows, OR opening an already-open conversation reuses its session so a second open never exists. The task notes say which approach was taken and why.
- [ ] #2 A regression test drives the second-open case through the production hydration path.
<!-- AC:END -->
