---
id: TASK-42550
title: "A captured manual switch from llama.cpp to a hosted provider with project instructions is refused"
status: To Do
assignee: []
created_date: 2026-11-25 17:35
labels:
- follow-up
- console
- trace
priority: low
references:
- tldw_chatbook/Chat/console_trace_service.py
- tldw_chatbook/Chat/console_trace_fallback_rewrite.py
- tldw_chatbook/Chat/console_trace_slot_restore.py
- Tests/Chat/test_console_failed_call_retry_ownership.py
- backlog/decisions/337-console-reference-backed-semantic-trace-ledger.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while doing TASK-42525 (round 26, 2026-11-25); the owner's third decision for that task left it out of scope.

**Why.** A user whose captured chat has project instructions (AGENTS.md) and who switches the session from a llama.cpp server to a hosted provider by hand gets "Trace capture could not start" on the next message (`unsupported_surface_change`); **Send without capture** sends it. llama.cpp sends the system prompt as the first message, so the trace surface has a leading system slot. A hosted provider takes the system prompt outside the message list, so its request has no row for that slot. ADR-337 lets only a provider fallback's first call remove the slot (amendment 2026-11-24); a manual switch is a new turn's first call, which may only put a slot back (amendment 2026-11-25), not take one away.

**Evidence.** A probe on fu/r26-trace2 (deleted after use) with the real controller, bridge, gateway and trace factory, only the provider handlers faked: first message to llama.cpp, then the session resolves to a hosted provider. With AGENTS.md the next message was refused (`['unsupported_surface_change']`, run `blocked`, the hosted handler never called); without project instructions it was sent.

**Decision needed.** Whether a new turn's first call may also remove the leading slot (the mirror of amendment 2026-11-25 for the other direction), and under which checks. That is an owner decision for ADR-337.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The owner's decision on removing the leading system slot at a manual switch is recorded in the task and, if granted, in ADR-337
- [ ] #2 If granted: with Capture On and project instructions, a manual switch from llama.cpp to a hosted provider is sent and traced, and the next message after it is sent too
- [ ] #3 A test drives that switch through the real controller; it is red before the change and green after
<!-- AC:END -->
