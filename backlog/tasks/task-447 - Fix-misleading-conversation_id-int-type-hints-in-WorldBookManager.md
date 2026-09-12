---
id: TASK-447
title: 'Fix misleading conversation_id: int type hints in WorldBookManager'
status: Done
assignee:
  - '@zcode'
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
- [x] #1 All conversation_id parameters on WorldBookManager conversation-facing methods are annotated conversation_id: str (not int),No runtime behavior change; existing world-book conversation-attach tests stay green,Docstrings/annotations consistent across associate/disassociate/get_world_books_for_conversation/get_conversations_for_world_book
<!-- AC:END -->

## Implementation Plan

1. Verify on current dev: which methods carry the wrong hint, that every production caller passes str(conversation_id), and that conversation_world_books.conversation_id is TEXT.
2. Change the annotations on all affected methods together; run the world-book manager and resolver test files.

ADR required: no
ADR path: N/A
Reason: Annotation-only correction; no runtime behavior change.

## Implementation Notes

Changed ``conversation_id: int`` to ``conversation_id: str`` on ``associate_world_book_with_conversation``, ``disassociate_world_book_from_conversation``, and ``get_world_books_for_conversation`` (tldw_chatbook/Character_Chat/world_book_manager.py). The fourth method named by the task, ``get_conversations_for_world_book``, takes only ``world_book_id`` and already documents its output as ``"conversation_id": str`` -- no wrong hint existed there. Verified the premise first: every production caller passes ``str(conversation_id)`` (chat_screen.py twice, personas_screen.py, world_info_resolver.py), and the junction column is ``conversation_id TEXT`` (ChaChaNotes_DB.py schema), so SQLite's dynamic typing made the old calls correct while the hints misled callers -- cosmetic, exactly as filed.

Verification: Tests/Character_Chat/test_world_book_manager.py 32 passed; with resolver and send-path files 44 passed. Ruff: zero delta (54 pre-existing fixables both sides). No runtime change; no docstring edits needed beyond the annotations (Args sections say "The ID of the conversation", type-neutral).

Modified: tldw_chatbook/Character_Chat/world_book_manager.py (three signatures).
