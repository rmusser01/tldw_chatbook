---
id: TASK-453
title: 'DocumentGenerator.get_conversation_context always returns empty (missing DB method)'
status: Done
assignee: []
created_date: '2026-07-21 20:20'
labels:
  - bug
  - document-generation
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`DocumentGenerator.get_conversation_context` (`Chat/document_generator.py:127-147`) calls `self.db.get_messages_by_conversation_id(...)`, a method that does not exist on `CharactersRAGDB` (nor any DB class in `DB/`). The `AttributeError` is caught, so the method silently returns empty context. Consequence: timeline / study-guide / briefing generation has been running with NO conversation context for every real invocation — the generated documents ignore the actual conversation content. Pre-existing (predates the internal-prompts migration); surfaced while writing P2 migration tests, which is why every doc-gen test sees `context == ""`. Fix by calling the correct DB accessor (likely `get_messages_for_conversation` or equivalent — verify the real method name on CharactersRAGDB) so context is actually populated.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 get_conversation_context returns the real messages for a conversation that has them (not empty)
- [x] #2 The correct existing DB accessor is used (no reliance on the nonexistent get_messages_by_conversation_id)
- [x] #3 A test covers timeline/study-guide/briefing generation with a NON-empty context reaching the LLM payload
- [x] #4 The empty/no-conversation case still degrades gracefully (no crash)
<!-- AC:END -->

## Implementation Notes

get_conversation_context now calls the real get_messages_for_conversation (was the nonexistent get_messages_by_conversation_id) and normalizes each row's `sender` to the `role` key format_context_for_llm reads (a second latent bug). Missing conversation → empty context, no crash. Two new tests: real-context-in-payload + graceful-empty.
