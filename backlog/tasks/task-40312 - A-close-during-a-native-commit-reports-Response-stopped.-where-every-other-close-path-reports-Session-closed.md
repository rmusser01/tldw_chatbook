---
id: TASK-40312
title: "A close during a native commit reports 'Response stopped.' where every other close path reports 'Session closed.'"
status: To Do
created_date: 2026-11-12 19:36
labels:
- follow-up
- console
- copy
priority: low
references:
- tldw_chatbook/Chat/console_native_commit.py
- tldw_chatbook/Chat/console_chat_controller.py
- Tests/Chat/test_console_durable_commit_window.py
dependencies:
- TASK-39155
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (cprod stream, TASK-39155). Checked by reading the code at the round-13 stack top 8b633d0101, 2026-11-12; line numbers are that tree's.

**Why.** The same user action, closing a chat mid-turn, reports two different outcomes depending on timing. That is the drift TASK-22690 set one name to prevent: `SESSION_CLOSED_COPY = "Session closed."` (`console_chat_controller.py:2392-2394`, "One name for it so the six sites that reported a close cannot drift apart").

**Evidence.** TASK-39155 made a close in progress count as explicit (`console_chat_controller.py:25672`). `console_native_commit.py:488-495` then sets `content = "Response stopped."` for an explicit stop, and the same text goes to the run state and the turn result (`:520-534`, `:563-568`, `:582-597`). The temporary-chat setup path ends "Session closed.". Both are pinned: `Tests/Chat/test_console_durable_commit_window.py:149` ("Response stopped.") and `:229` ("Session closed."). The persisted reply text can stay "Response stopped."; the run state and turn result are the close's report.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A close during a native commit ends the turn with SESSION_CLOSED_COPY and a stopped state, with no blocker
- [ ] #2 The durable-commit tests pin the same copy for both close windows; one is red before the fix
<!-- AC:END -->
