---
id: TASK-453
title: 'DocumentGenerator.get_conversation_context always returns empty (missing DB method)'
status: To Do
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
- [ ] #1 get_conversation_context returns the real messages for a conversation that has them (not empty)
- [ ] #2 The correct existing DB accessor is used (no reliance on the nonexistent get_messages_by_conversation_id)
- [ ] #3 A test covers timeline/study-guide/briefing generation with a NON-empty context reaching the LLM payload
- [ ] #4 The empty/no-conversation case still degrades gracefully (no crash)
<!-- AC:END -->
