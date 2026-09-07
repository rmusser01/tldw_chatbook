---
id: TASK-447
title: 'Fix misleading conversation_id: int type hints in WorldBookManager'
status: To Do
assignee: []
created_date: '2026-07-21 15:06'
labels:
  - lore
  - tech-debt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
WorldBookManager's conversation-facing methods (associate_world_book_with_conversation, disassociate_world_book_from_conversation, get_world_books_for_conversation, get_conversations_for_world_book) annotate conversation_id: int, but the runtime value is always a string UUID and the conversation_world_books.conversation_id column is TEXT. SQLite's dynamic typing makes current calls (which pass str(...)) correct, so this is cosmetic, but the hints mislead callers. Gemini flagged the associate/disassociate pair on PR #738 (P2g-2); fixing them piecemeal would be inconsistent with the sibling getters, so do all of them together.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All conversation_id parameters on WorldBookManager conversation-facing methods are annotated conversation_id: str (not int),No runtime behavior change; existing world-book conversation-attach tests stay green,Docstrings/annotations consistent across associate/disassociate/get_world_books_for_conversation/get_conversations_for_world_book
<!-- AC:END -->
