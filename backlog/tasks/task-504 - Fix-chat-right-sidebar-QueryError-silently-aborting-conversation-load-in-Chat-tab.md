---
id: TASK-504
title: >-
  Fix #chat-right-sidebar QueryError silently aborting conversation load in Chat
  tab
status: To Do
assignee: []
created_date: '2026-07-24 01:29'
labels:
  - tech-debt
  - dead-code
  - chat
  - bug
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
display_conversation_in_chat_tab_ui() (Event_Handlers/Chat_Events/chat_events.py) queries app.query_one("#chat-right-sidebar") unconditionally while populating a loaded conversation into the Chat tab. That id has not existed in the live compose tree since ChatWindowEnhanced replaced the legacy ChatWindow (right sidebar functionality moved into settings_sidebar) -- discovered during task-412's chat_right_sidebar.py deletion audit. The query is wrapped by a broad try/except QueryError that swallows the exception and shows a generic 'Error updating UI for loaded chat.' notification, but because the query sits partway through the population logic, everything after it (conversation title, keywords, UUID display, and the full message log) never gets populated when it fires. Users loading a saved conversation in the live Chat tab may be seeing an empty/stale chat log today.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Loading a saved conversation in the live Chat tab (ChatWindowEnhanced) populates the title, keywords, UUID display, and full message log without hitting the #chat-right-sidebar QueryError path
- [ ] #2 The character-detail-edit fields this function tries to populate (chat-character-name-edit etc.) are either restored somewhere reachable in the live UI, or the dead write attempt is removed without regressing conversation loading
- [ ] #3 A regression test loads a conversation through display_conversation_in_chat_tab_ui (or its tab-aware wrapper) against a live-shaped widget tree (no #chat-right-sidebar) and asserts the message log and title actually populate
<!-- AC:END -->
