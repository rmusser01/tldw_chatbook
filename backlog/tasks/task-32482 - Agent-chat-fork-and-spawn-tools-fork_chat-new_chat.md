---
id: TASK-32482
title: Agent chat fork and spawn tools (fork_chat / new_chat)
status: To Do
assignee: []
created_date: '2026-09-12 01:25'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let a Console agent prepare parallel-workstream chats for the user: fork_chat copies the current conversation's active path into a new chat (first-ever writer of the parent_conversation_id / forked_from_message_id lineage columns); new_chat creates a fresh chat. Both accept title / opening_prompt / instructions, require an explicit user approval per call by default (Allow / Allow-for-session / Deny, fail-closed, session-scoped remember), land the opening prompt as a composer draft the user sends, and open the chat in the background with a toast. Spec: Docs/superpowers/specs/2026-09-11-agent-chat-fork-spawn-design.md; ADR: backlog/decisions/150-agent-chat-fork-and-spawn.md; Plan: Docs/superpowers/plans/2026-09-11-agent-chat-fork-spawn.md; follow-ups: sub-agents (task-32480) and preset/provider args (post ADR-147 integration PR).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Agent can fork the current chat into a new one via a confirmed tool call (verbatim active-path copy with lineage columns set)
- [ ] #2 Agent can create a fresh chat with title and instructions via a confirmed tool call
- [ ] #3 Every creation requires explicit user approval by default with per-tool session-scoped remember and fail-closed behavior when no UI or on cancel or timeout
- [ ] #4 Opening prompt lands as a composer draft the user sends and survives app restart via conversation metadata
- [ ] #5 New chats appear in the same workspace without switching the active session and a toast announces them
- [ ] #6 Forks of character-bound chats refuse agent instructions and ephemeral or empty sources return clear tool errors
- [ ] #7 Tool schemas are advertised to primary agents only while sub-agent runs are unchanged
- [ ] #8 Workspace listing shows created chats under whichever mechanism governs it
- [ ] #9 Tests cover fork helper semantics (remap, lineage, atomicity, uncapped copy) and confirm rounds (allow, deny, remember, fail-closed, park)
- [ ] #10 User docs updated for both tools
<!-- AC:END -->
