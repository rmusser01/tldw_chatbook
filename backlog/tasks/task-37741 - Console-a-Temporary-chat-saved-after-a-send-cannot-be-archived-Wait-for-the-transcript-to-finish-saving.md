---
id: TASK-37741
title: >-
  Console: a Temporary chat saved after a send cannot be archived ('Wait for the
  transcript to finish saving')
status: To Do
assignee: []
created_date: '2026-10-29 16:09'
labels:
  - console
  - tabs-sessions
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found live 2026-10-29 while verifying TASK-33621.17, and reproduced on dev 38c9d2edcf (pre-existing). Open a Temporary tab, send one message and get the reply (dev: Send without capture on the pause card; after TASK-33621.17 the default send), then save the chat (status-strip chip or composer Menu 'Save chat'). 'Archive this chat' then refuses with 'Wait for the transcript to finish saving before archiving.' and keeps refusing; a chat that was never temporary archives fine in the same session. conversation_archive_actions.py:135-142 refuses while any session message is in the store's _pending_persistence_message_ids or has no persisted_message_id, so promotion apparently leaves one transcript row of the temporary turn without its persisted id. TASK-33621.17 makes 'send, then Save chat' the main Temporary path, so this now meets every Temporary chat that is kept.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A Temporary chat with at least one answered turn, once saved, can be archived
- [ ] #2 A test drives send, reply, Save chat and Archive through the real store and fails before the fix
<!-- AC:END -->
