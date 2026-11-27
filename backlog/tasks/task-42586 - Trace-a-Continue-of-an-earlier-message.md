---
id: TASK-42586
title: "Trace a Continue of an earlier message"
status: To Do
assignee: []
created_date: 2026-11-26 19:41
labels:
- follow-up
- console
- trace
priority: medium
references:
- tldw_chatbook/Chat/console_action_capture.py
- tldw_chatbook/Chat/console_trace_action_calls.py
- tldw_chatbook/Chat/console_trace_rewritten_tail.py
- Tests/Chat/test_console_trace_action_calls.py
- backlog/decisions/337-console-reference-backed-semantic-trace-ledger.md
dependencies:
- TASK-42560
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in TASK-42560's review (round 27, 2026-11-26).

**Why.** Continue is offered on any message. Its request runs only through the selected message, but its reply is appended at the end of the chat. Traced as a call of its turn, it would re-record the selected turn and drop the later turns, and the next message was then refused (`unsupported_surface_change`) even without project instructions. TASK-42560 therefore keeps a Continue of a message with later rows after it Capture Off, as before (ADR-337 amendment 2026-11-26, point 1).

With project instructions (AGENTS.md), the message after such a Capture Off Continue is still refused ("Trace capture could not start"), as it was before TASK-42560, for both an earlier reply and an earlier user message (`test_a_continue_of_an_earlier_message_keeps_capture_off[*-agents_md]` pins it). Without them it is sent.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With Capture On, a Continue of an earlier reply and of an earlier user message is traced as what it sent, with or without project instructions, shown by a test through the real controller with only the provider faked, red before and green after
- [ ] #2 The message after it, and the one after that, are sent and traced, with or without project instructions
- [ ] #3 Any new admission rule is recorded in ADR-337 before the code
<!-- AC:END -->
