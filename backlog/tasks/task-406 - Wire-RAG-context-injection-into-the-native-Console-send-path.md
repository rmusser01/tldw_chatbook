---
id: TASK-406
title: Wire RAG context injection into the native Console send path
status: To Do
assignee: []
created_date: '2026-07-21 09:48'
updated_date: '2026-08-07 14:20'
labels:
  - rag
  - console
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verified during the RAG scope program (plan verification V4, re-confirmed by two independent reviews): the native Console send path (ConsoleChatController.submit_draft) performs NO RAG context injection at all — get_rag_context_for_chat's only production caller is the legacy chat_events send path, which is unreachable in the live app (routes/dead-sites traced in the Task 5 review). Users of the native Console therefore get no chat-RAG injection regardless of settings. Mirror the chat-dictionaries precedent (PR 664: transform applied at all native send sites): resolve scope + inject RAG context in the native path, honoring conversation scope (Chat/rag_scope.py resolution seams exist and are public as resolve_effective_scope_for_chat).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Native Console sends inject RAG context only when the 'Auto-retrieve on send' toggle in the Console RAG chip modal is ON (default OFF; enablement is re-homed from the legacy sidebar checkbox to this modal toggle, persisted as a global config key); when ON, injection routes through the existing staged-evidence pipeline -- a visible 'Evidence sent · N' strip, consumed only on send -- honoring conversation scope end-to-end, never as invisible prompt injection.
- [ ] #2 EMPTY scope short-circuits with the shared notice copy on the native path
- [ ] #3 Legacy path behavior unchanged
<!-- AC:END -->
