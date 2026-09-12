---
id: TASK-32379
title: A conversation you never titled shows the stored literal 'New Chat' in Library
status: To Do
assignee: []
created_date: '2026-09-11 08:48'
labels:
  - library
  - console
  - conversations
  - copy
  - critique-10
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #10 named 'New Chat' as schema-speak in the Library conversations list. It is not a Library display fallback -- library_conversations_state already renders a BLANK title as 'Untitled conversation'. 'New Chat' is a real stored title written at creation time by Chat/chat_conversation_service.py and Chat/chat_persistence_service.py, so relabelling it in the Library would also relabel a conversation a user deliberately named that. Deciding what an untitled conversation is called, and where that name is decided, is a cross-surface product call. Evidence: critique #10 fix-wave review of task-32364, finding 8.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An untitled conversation reads the same way in Console and in Library
- [ ] #2 A conversation a user deliberately titled 'New Chat' keeps that title
<!-- AC:END -->
