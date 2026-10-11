---
id: TASK-33667
title: Settle a dispatch recovery from a second open of the same chat
status: Done
assignee:
  - '@claude'
created_date: '2026-10-02 19:50'
updated_date: '2026-10-11 02:20'
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
- [x] #1 With the same chat open twice after a relaunch, Retry and Discard settle the recovery from either open, without changing the other session's rows, OR opening an already-open conversation reuses its session so a second open never exists. The task notes say which approach was taken and why.
- [x] #2 A regression test drives the second-open case through the production hydration path.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Find every route that hydrates a saved conversation and which of them can run while that conversation is already open.
2. Choose between reusing the open session and mapping recovery owners per session.
3. Write a test that opens the same chat twice through a mounted Console over a database copied mid-reply; watch it fail.
4. Make the open path reuse the session.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Opening a conversation that is already open now activates its session. A second session for it is never hydrated, so no card is left that cannot settle.

**Approach: reuse the open session** (AC#1's second branch). `UI/Console_Modules/workspace.py::_resume_console_workspace_conversation` (the function this rider calls `open_console_workspace_conversation_by_id`) now defaults to `reuse_existing=True`. After the saved record loads, a session whose `persisted_conversation_id` matches is activated through `open_console_workspace_conversation("native:<id>")`; only a conversation with no open session is hydrated.

**Why not a per-session owner map.**
- The map would have to be honoured by every recovery path the rider lists (claim, retry, discard, release, settle, the generation-token bind, the continuation handoff, and the controller's `before_message_id` and `get_message` reads), in `console_chat_store.py` and the controller, to support a state no route needs.
- Every other route already reused. Both rail-row callers look the session up first (`session.py::_console_session_id_for_browser_row`, `workspace.py::_console_session_id_for_workspace_conversation`). The switcher passes `reuse_existing=True` and Library's restore passes `preserve_persisted_scope=True`. The launch wake (`console_launch_wake.py`) and the buddy restore (`buddy_conversation.py`) skip a conversation that is already open. This default was the one seam left: two opens racing through the tree load, or a caller that skips its own lookup, got a second session.
- The one row shape that asked for a second session on purpose cannot occur. `_console_session_id_for_browser_row` returns no session for a "membership" row whose workspace differs from the open session's, but `_membership_console_browser_rows`, the only producer of such rows, has no caller in `tldw_chatbook` (one definition, no call).

The check runs after the tree-load await and immediately before `hydrate_console_session`, which registers the session in the store before its own first await. Two opens of the same chat therefore cannot both hydrate.

The store keeps its own guard for a second session built some other way: `test_a_second_open_of_the_same_chat_never_settles_the_first_ones_reply` still passes, so such a session's card refuses instead of claiming the first session's rows.

**Ruling.** The default is flipped; the parameter stays — the switcher in `UI/Console_Modules/session.py` passes it, and Phase 8 edits that file in this wave; nothing passes `False` — cost if wrong: a later caller that passes `False` gets a second session whose card refuses, as every such session did before this change.

**Test (AC#2).** `Tests/Chat/test_console_dispatch_recovery_relaunch.py::test_opening_an_open_chat_again_reuses_its_session_so_the_recovery_settles`. A real controller sends a turn whose provider hangs, and the SQLite file is copied at that moment. A mounted `ChatScreen` over that copy, with the real `ChatConversationScopeService`, then runs the production open path (the tree load, then `hydrate_console_session`). The test opens the chat, switches to another session and opens the chat again with the rail row's call (no flag). It asserts one session for the conversation, active again, and that Discard settles the recovery from it. Before the change it failed: the second open added a second session.

No User Guide change: the routes a user can reach already reused the tab (sessions-tabs-workspaces.md says so for Library's restore).

Files: `tldw_chatbook/UI/Console_Modules/workspace.py`, `Tests/Chat/test_console_dispatch_recovery_relaunch.py`.
<!-- SECTION:NOTES:END -->
