---
id: TASK-42565
title: "A trace reservation refused at dispatch binding is left reserved instead of settled"
status: To Do
assignee: []
created_date: 2026-11-26 07:39
labels:
- follow-up
- console
- trace
priority: low
references:
- tldw_chatbook/Chat/console_trace_service.py
- tldw_chatbook/Chat/console_provider_gateway.py
- backlog/decisions/337-console-reference-backed-semantic-trace-ledger.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Raised by the round-27 review of TASK-42556 (2026-11-26).

**What happens.** A Capture-On call is reserved in one transaction and bound at dispatch in another (`ConsoleProviderGateway._commit_trace_dispatch_started` → `ConsoleTraceService.persist_request` with `reserved_call`). When the binding transaction refuses the call (for example, the failed-call retry comparison in `persist_request` raising `trace_tool_chain_unavailable`), the binding rolls back, the gateway raises `TraceCallPersistenceError`, and nothing settles the reservation. The call stays `reserved`, with no route identity, until startup recovery maps it to `not_dispatched`.

**Where it was seen.** On dev 6e41919880, a captured spawned child's preset fallback was reserved as an ordinary retry of its failed call and refused at binding (TASK-42556's notes). The child's segment kept a `reserved` row; the provider was never asked. TASK-42556 and TASK-42564 removed that trigger, but any other binding refusal still leaves a row. TASK-42564's own binding recheck is one, and its test pins today's `reserved` result.

**Why it matters.** Readers of the ledger treat a `reserved` call as unsettled. The closed-run witness and `has_unresolved_call_for_turn` (for an owner's own segments) refuse or wait on it, and recovery reports it after a restart, although the request was never sent.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A call refused at dispatch binding ends in a settled state that says it was not dispatched, without a restart
- [ ] #2 A turn whose call was refused at binding is not reported as having an unsettled call, and the next message is not blocked by it
- [ ] #3 A test drives a binding refusal through the real gateway and trace factory, red before and green after, on a PR lane
<!-- AC:END -->
