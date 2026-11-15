---
id: TASK-41048
title: "Retire the legacy sidebar RAG entry points that nothing in the app calls"
status: To Do
created_date: 2026-11-15 02:03
labels:
- follow-up
- architecture-review-2026-10-25
- cleanup
- rag
priority: low
references:
- tldw_chatbook/Event_Handlers/Chat_Events/chat_rag_events.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Event_Handlers/Chat_Events/chat_rag_events.py` still holds four entry points with no production caller on dev ecad404bd4 (2026-11-15); only tests reach them (found by TASK-36947.3):
- `get_rag_context_for_chat` and `get_rag_context_capture_for_chat`, which read the old `#chat-rag-*` sidebar widgets;
- the two app-taking wrappers `authorize_local_results_for_prompt` and `assemble_local_evidence_for_prompt`.

Dead entry points look live, they keep app-shaped signatures alive, and their tests cost upkeep.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The four functions are deleted, or the task notes name the production caller that keeps each one
- [ ] #2 Their tests are deleted or retargeted to the live RAG_Search functions
- [ ] #3 No test or production import breaks
<!-- AC:END -->
