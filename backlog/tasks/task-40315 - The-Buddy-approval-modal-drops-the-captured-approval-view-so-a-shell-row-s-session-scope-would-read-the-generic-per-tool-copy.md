---
id: TASK-40315
title: "The Buddy approval modal drops the captured approval view, so a shell row's session scope would read the generic per-tool copy"
status: Done
assignee: ['@claude']
updated_date: 2026-11-13 02:48
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
- [x] #1 The notes say whether a raw-shell call can reach the Buddy approval modal, with the evidence
- [x] #2 If it can, Buddy passes the captured view (or a shell row there states the shell scope), and a test pins the copy; if it cannot, the notes name the guard that prevents it and the task closes
<!-- AC:END -->

## Implementation Plan

1. Trace whether a raw-shell approval round can reach `BuddyConversationModal` on base 26b2e677dd, and look for any guard.
2. If it can, drive it end to end with a test (real controller, real Buddy coordinator, real raw-shell row) that pins the shell session scope; show it red.
3. Pass the payload's captured `view` (and its presentation revision/state, as Console's `ChatTaskCards._set_approval` does) to `set_batch` in the modal.

## Implementation Notes

**It can reach Buddy; no guard prevents it.** The premise held on base 26b2e677dd (`buddy_conversation_modal.py:288-294` passed no `view=`). The evidence:
- `BuddyBinding.for_session` binds any local Console conversation (`Persona_Buddy/interaction.py:50-62`).
- `BuddyConversationCoordinator.decision_payloads` returns that session's `controller.pending_decision_projection(session.id)` (`UI/Navigation/buddy_conversation.py:269-283`). Its `_pending_decision_payloads_locked` filters on `session_id` only (`console_hook_review_host.py:50-64`), with no provider or kind filter.
- The raw-shell provider is composed per Console session and asks through `request_mcp_approvals(session_id=...)` (`console_provider_composition.py:897-913`), so its round is an ordinary approval payload carrying a captured `view` with `grant_domain="console_chat"`.

The new test proves the same path running. `test_buddy_raw_shell_round_states_the_shell_session_scope` (in `Tests/UI/test_console_mcp_approval.py`, a whole-file UI census entry) uses `Harness` from `test_buddy_conversation_modal.py`: a real `ConsoleChatController`, a real `BuddyConversationCoordinator` through `open_buddy_conversation`, and a row from the real `RawShellToolProvider.pending_gate_for` raised with `request_mcp_approvals`. The Buddy card renders the round. With "Until Chatbook exits" chosen, its scope line read "Every call to this tool until Chatbook exits." on base (red, `-n 4`).

**Fix.** The modal now passes `view`, `presentation_revision` and `presentation_state` from the payload, as Console's `ChatTaskCards._set_approval` does. The same test now reads "Until Chatbook exits: future shell commands in this Console chat. Clears on Disarm or when Chatbook exits." (green).

**A trap the test hit.** A row the card has mounted can hold a decision Select that has not mounted yet (`value` is `Select.NULL`). A choice set then landed, but never reached the scope line, and the first red read the Deny copy for that reason. The shared helper `_session_scope_line` now waits, bounded, for the Select to mount before choosing.

Covering run (`-n 4`, load 62): `Tests/UI/test_buddy_conversation_modal.py`, `Tests/Persona_Buddy/test_buddy_modal_poll_gating.py`, `Tests/UI/test_buddy_management_journey.py` and `Tests/UI/test_buddy_v1_qualification_capture.py`: 76 passed, 1 failed. The failure, `test_cold_home_saved_buddy_bootstraps_only_after_explicit_open`, fails the same way with the base modal (a child process's 5 s `until` deadline under load). The User Guide quotes no scope copy for Buddy. `console/agent-runs-and-tools.md` already says raw-shell grants cover the Console chat, which the modal now matches.

Files: `tldw_chatbook/Widgets/Persona_Widgets/buddy_conversation_modal.py`, `Tests/UI/test_console_mcp_approval.py`.
