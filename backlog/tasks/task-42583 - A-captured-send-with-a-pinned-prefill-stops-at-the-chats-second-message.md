---
id: TASK-42583
title: "A captured send with a pinned prefill stops at the chat's second message"
status: To Do
assignee: []
created_date: 2026-11-26 18:48
labels:
- follow-up
- console
- trace
priority: medium
references:
- tldw_chatbook/Chat/console_trace_service.py
- tldw_chatbook/Chat/console_action_capture.py
- backlog/decisions/337-console-reference-backed-semantic-trace-ledger.md
dependencies:
- TASK-42560
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while doing TASK-42560 (round 27, 2026-11-26).

**Why.** In a captured chat with a pinned response prefill, the second message stops at "Trace capture could not start" (`surface_prefix_mismatch`); the user can only send it with **Send without capture**. A scratch probe through the real controller (a pinned prefill sends on the DIRECT_PREFILL route; llama.cpp HTTP faked) showed this on base 7ebae7bd52, before TASK-42560: the first send was traced, and the second was refused before dispatch.

Because of this, TASK-42560 left the actions that send the pinned prefill (Edit & resend, Regenerate and Resend) sending with Capture Off, as before (ADR-337 amendment 2026-11-26, point 1). A plain Continue sends no prefill and is traced. Once a prefill chat's sends are traced, those actions should be traced too.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With Capture On and a pinned response prefill, every message of the chat is sent and traced as sent, shown by a test through the real controller with only the provider faked, red before and green after
- [ ] #2 With a pinned prefill, Edit & resend, Regenerate and Resend (which send it) are traced as calls of their turn, and the message after each is sent and traced
- [ ] #3 Any new admission rule is recorded in ADR-337 before the code
<!-- AC:END -->
