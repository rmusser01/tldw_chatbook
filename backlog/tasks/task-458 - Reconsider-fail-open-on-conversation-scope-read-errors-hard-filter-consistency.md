---
id: TASK-458
title: >-
  Reconsider fail-open on conversation-scope read errors (hard-filter
  consistency)
status: To Do
assignee: []
created_date: '2026-07-22 03:51'
labels: []
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR #757 review (comment 5) made the workspace-scope read in resolve_scope_for_session (tldw_chatbook/Event_Handlers/Chat_Events/chat_rag_events.py) fail CLOSED on a registry read failure -- returning an EMPTY EffectiveScope rather than silently dropping the workspace bound. The pre-existing conversation-scope read in the same function (read_conversation_scope, wrapping tldw_chatbook/Chat/rag_scope.py's Phases 1-2 storage read) still fails OPEN: any exception there is left uncaught by resolve_scope_for_session's own try/except (it only wraps the workspace read), so a conversation-scope storage read failure either propagates or, depending on the caller, can be swallowed upstream and treated as conv_scope=None -- i.e. 'no scope', which then lets retrieval proceed unrestricted (or bounded only by an intersecting workspace scope) instead of failing closed. For a hard-filter retrieval-scope feature, an unreadable stored scope should arguably never be indistinguishable from a deliberately-unset one. Evaluate whether the conversation-scope read path should adopt the same fail-closed-to-EMPTY posture the workspace-scope read now has, for consistency and to close this asymmetry.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Investigation documents how a conversation-scope read failure currently propagates through resolve_scope_for_session and get_rag_context_for_chat (raises vs. silently degrades to unscoped)
- [ ] #2 A decision is recorded: either the conversation-scope read is changed to fail closed (EMPTY, matching the PR#757 comment-5 workspace-scope precedent) with tests, or the fail-open behavior is deliberately kept with documented rationale
- [ ] #3 If changed, existing conversation-scope tests in Tests/RAG/test_scope_pipeline_enforcement.py and Tests/Chat/test_rag_scope.py continue to pass and new tests cover the failure path
<!-- AC:END -->
