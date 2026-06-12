---
id: TASK-95
title: Bind character sessions for start-chat handoffs in Console
status: To Do
assignee: []
created_date: '2026-06-11 04:54'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Personas Start Chat stages a handoff with metadata intent=start_chat, but chat_screen._session_data_for_handoff does not set character_id/assistant_kind, so the session is not character-bound (no greeting machinery). Teach the Console handoff consumer to bind the character when metadata.selected_kind == character.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Start Chat from Personas opens a session with the character bound,Greeting/first-message machinery applies
<!-- AC:END -->
