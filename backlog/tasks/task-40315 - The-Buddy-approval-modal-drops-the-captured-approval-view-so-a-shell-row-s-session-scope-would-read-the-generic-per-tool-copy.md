---
id: TASK-40315
title: "The Buddy approval modal drops the captured approval view, so a shell row's session scope would read the generic per-tool copy"
status: To Do
created_date: 2026-11-12 19:36
labels:
- follow-up
- console
- approvals
- copy
priority: low
references:
- tldw_chatbook/Widgets/Persona_Widgets/buddy_conversation_modal.py
- tldw_chatbook/Chat/approval_presentation.py
- tldw_chatbook/Widgets/Chat_Widgets/chat_approval_card.py
dependencies:
- TASK-40097
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (cred stream review). Checked by reading the code at the round-13 stack top 8b633d0101, 2026-11-12; line numbers are that tree's.

**Why.** The scope line is the user's consent statement. `BuddyConversationModal` calls `set_batch` without `view=` (`buddy_conversation_modal.py:288-294`), although the payload carries one (`approval_presentation.py:367`). A Buddy shell row on approve_session would therefore read "Every call to this tool until Chatbook exits." (`chat_approval_card.py:129`) instead of the chat-scoped shell scope. Whether a raw-shell call can reach Buddy at all is not verified.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The notes say whether a raw-shell call can reach the Buddy approval modal, with the evidence
- [ ] #2 If it can, Buddy passes the captured view (or a shell row there states the shell scope), and a test pins the copy; if it cannot, the notes name the guard that prevents it and the task closes
<!-- AC:END -->
