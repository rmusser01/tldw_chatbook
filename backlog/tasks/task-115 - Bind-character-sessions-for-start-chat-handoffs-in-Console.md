---
id: TASK-115
title: Bind character sessions for start-chat handoffs in Console
status: Done
assignee: []
created_date: '2026-06-11 04:54'
updated_date: '2026-06-17 20:12'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Personas Start Chat stages a handoff with metadata intent=start_chat, but chat_screen._session_data_for_handoff does not set character_id/assistant_kind, so the session is not character-bound (no greeting machinery). Teach the Console handoff consumer to bind the character when metadata.selected_kind == character.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Start Chat from Personas opens a session with the character bound,Greeting/first-message machinery applies
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: bounded Console handoff mapping bugfix; no storage, sync, runtime, or cross-module ownership decision changes.

1. Add a focused regression for a Personas `intent=start_chat` character payload converted by `ChatScreen._session_data_for_handoff()`.
2. Verify the regression fails because the session lacks `character_id`, `character_name`, `assistant_kind`, and `assistant_id`.
3. Add a small metadata parser/helper in `chat_screen.py` so character start-chat handoffs bind the session identity.
4. Keep non-character and non-start-chat handoffs unchanged.
5. Rerun the focused handoff tests plus the Personas Start Chat producer test.
6. Update TASK-115 acceptance criteria and implementation notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a focused Console handoff regression for Personas `intent=start_chat` character payloads. `ChatScreen._session_data_for_handoff()` now binds character Start Chat handoffs into `ChatSessionData` with `character_id`, `character_name`, `assistant_kind="character"`, and `assistant_id`, while leaving generic/attach handoffs unchanged. This gives downstream Console tab title, persistence, and greeting machinery the character identity it expects.

Verification:
- `python -m pytest -q Tests/UI/test_chat_first_handoffs.py::test_chat_screen_start_chat_handoff_binds_character_session_identity --tb=short`
- `python -m pytest -q Tests/UI/test_chat_first_handoffs.py --tb=short`
- `python -m pytest -q Tests/UI/test_personas_workbench.py::TestConsoleActions::test_start_chat_uses_real_mechanism --tb=short`
- `git diff --check`
<!-- SECTION:NOTES:END -->
