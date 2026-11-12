---
id: TASK-40313
title: "A refused automatic or manual mixin attempt leaves an empty accepted reply row that reads as a pending response"
status: To Do
created_date: 2026-11-12 19:36
labels:
- follow-up
- console
- persistence
priority: medium
references:
- tldw_chatbook/Chat/console_prompt_mixin_service.py
- tldw_chatbook/Chat/chat_persistence_service.py
- tldw_chatbook/Chat/console_conversation_hydration.py
- tldw_chatbook/Chat/assistant_generation_state.py
dependencies:
- TASK-39159
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (cprod stream, TASK-39159). Checked by reading the code at the round-13 stack top 8b633d0101, 2026-11-12; line numbers are that tree's.

**Why.** A refused turn should leave no trace. Today it leaves an empty assistant row in the DB, and reopening the conversation probably shows it as "Response accepted; waiting for dispatch." (inferred from the code, not run), which reads as a send that is still pending or stuck.

**Evidence.** `console_prompt_mixin_service.py:695-705` raises `mixin_attempt_owner_changed` after the save and before `adopt_saved_pending_assistant`. The saved row (`chat_persistence_service.py:1302-1325`) is empty with `assistant_generation_state='accepted'`. The manual path, `accept_attempt` (`:593-604`), leaves the same row. Hydration keeps any row whose generation state is set (`console_conversation_hydration.py:258-263`), and the copy for an accepted row is at `assistant_generation_state.py:25`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After a refused automatic or manual attempt, the saved reply row is settled to a terminal state or removed
- [ ] #2 Reopening the conversation shows no pending reply for the refused attempt
- [ ] #3 One test covers both refusal paths and is red before the fix
<!-- AC:END -->
