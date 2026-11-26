---
id: TASK-42550
title: "A captured manual switch from llama.cpp to a hosted provider with project instructions is refused"
status: In Progress
assignee:
- '@claude'
created_date: 2026-11-25 17:35
updated_date: 2026-11-26 04:57
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
- Tests/Chat/test_console_trace_slot_drop.py
- backlog/decisions/337-console-reference-backed-semantic-trace-ledger.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while doing TASK-42525 (round 26, 2026-11-25); the owner's third decision for that task left it out of scope.

**Why.** A user whose captured chat has project instructions (AGENTS.md) and who switches the session from a llama.cpp server to a hosted provider by hand gets "Trace capture could not start" on the next message (`unsupported_surface_change`); **Send without capture** sends it. llama.cpp sends the system prompt as the first message, so the trace surface has a leading system slot. A hosted provider takes the system prompt outside the message list, so its request has no row for that slot. ADR-337 lets only a provider fallback's first call remove the slot (amendment 2026-11-24); a manual switch is a new turn's first call, which may only put a slot back (amendment 2026-11-25), not take one away.

**Evidence.** A probe on fu/r26-trace2 (deleted after use) with the real controller, bridge, gateway and trace factory, only the provider handlers faked: first message to llama.cpp, then the session resolves to a hosted provider. With AGENTS.md the next message was refused (`['unsupported_surface_change']`, run `blocked`, the hosted handler never called); without project instructions it was sent.

**Decision needed.** Whether a new turn's first call may also remove the leading slot (the mirror of amendment 2026-11-25 for the other direction), and under which checks. That is an owner decision for ADR-337.

**Owner decision (2026-11-26).** The question asked was: "Should a new turn's first call be allowed to drop that leading system slot (the mirror of the 2026-11-25 slot-restore amendment, with the same checks and the existing removal marker)?" The owner answered: "Allow it (Recommended)". It is written into ADR-337 as the owner-approved amendment of 2026-11-26, after the 2026-11-25 slot-restore amendment.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The owner's decision on removing the leading system slot at a manual switch is recorded in the task and, if granted, in ADR-337
- [ ] #2 If granted: with Capture On and project instructions, a manual switch from llama.cpp to a hosted provider is sent and traced, and the next message after it is sent too
- [ ] #3 A test drives that switch through the real controller; it is red before the change and green after
<!-- AC:END -->

## Implementation Plan

1. Record the owner's decision (above) and write it into ADR-337 as the 2026-11-26 amendment, before any code.
2. Red first: a new test file `Tests/Chat/test_console_trace_slot_drop.py` (real controller, bridge, gateway and trace factory; only the provider handlers faked) with the probe shape (llama.cpp first message with AGENTS.md, then a hosted provider: that message and the next are sent and traced), the no-AGENTS.md control, switching back to llama.cpp, Compact now and Edit & resend after a drop, and a restart.
3. Planner: beside `restored_slot_projection`, a `dropped_slot_projection` in `Chat/console_trace_slot_restore.py` returns the surface without its leading rendered-system slot when the call is a new turn's first (AGENT_FIRST/FRESH, no fallback parent, no live-native custody), the slot is the surface's leading active entry, and the request has no leading system row. Every other row is then planned by the existing rules, on a cold parent whose rows follow the removal marker's sequence.
4. Binding: recheck (still the turn's first call; the slot still leads the surface; the request still has no leading system row) and write the existing content-free `surface_removal` marker (`TraceOmissionReason.SURFACE_SLOT_REMOVED`) before the call's other rows; no schema change. Keep the service diff small (PR #3717 also edits it).
5. Hard negatives at gateway level, each pinned by its refusal code, with a mutation of each guard: a non-first call, a request that keeps its system row, a leading row that is not a slot, a fallback or retry route, and a binding recheck that finds the surface changed.
6. Lane the new file (recovered-suites preferred), time it with the step's flags, update the lane contract tests; update the User Guide; raise any size ratchet by the exact growth; run the brief's checks.
