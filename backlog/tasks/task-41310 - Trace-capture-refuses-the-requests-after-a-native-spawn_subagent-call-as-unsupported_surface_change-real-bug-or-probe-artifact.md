---
id: TASK-41310
title: "Trace capture refuses the requests after a native spawn_subagent call as unsupported_surface_change: real bug or probe artifact?"
status: In Progress
created_date: 2026-11-16 19:55
labels:
- follow-up
- console
- agents
- tracing
priority: medium
references:
- tldw_chatbook/Chat/console_trace_service.py
- tldw_chatbook/Chat/console_trace_errors.py
- Tests/Chat/test_console_default_send_capture.py
- https://github.com/rmusser01/tldw_chatbook/pull/3548
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the 2026-11-13..16 burn-down. TASK-40715's real-app probe found it (PR #3548 lists it under "Follow-ups to be filed"). Checked on dev 5e3c8b2bd4 (2026-11-16) by reading the code; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/`. It was not re-run.

**Why.** In a headless real-app probe (the private-child harness in `Tests/Chat/test_console_default_send_capture.py`), a native `spawn_subagent` call was followed by two model requests: the child's, and the parent's follow-up. Trace capture refused both with `TraceSurfaceChangeRefused('unsupported_surface_change')`. If this is real, every sub-agent turn with trace capture on stops at "Trace capture blocked". The probe's provider response was fake, so the refusal may instead come from a reply shape no real provider sends. Nobody has yet determined which.

**Evidence.**
- `Chat/console_trace_errors.py:39-43`: `TraceSurfaceChangeRefused` is raised by a surface fence before any call is reserved.
- `Chat/console_trace_service.py:2766-2795`: the surface check admits only appended messages before an unchanged continuation, or a bounded rewritten tail. Anything else is refused as `unsupported_surface_change`, recorded as `count_or_span` or `sequence_gap`.
- Related, not the same: TASK-37602 covers the same refusal after Edit & resend on providers that record continuation groups.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A reproduction with a real provider adapter, or a provider double whose native tool-call response matches a real provider's, records whether the child's request and the parent's follow-up after a native `spawn_subagent` call are refused, and which refusal reason (`count_or_span` or `sequence_gap`) fires
- [ ] #2 If they are refused, both requests reach the provider with trace capture intact, pinned by a regression test through the real controller, store and trace runtime that is red on dev
- [x] #3 If they are not refused, the task notes name the probe's response shape that caused the refusal, and the probe harness is corrected
<!-- AC:END -->

## Implementation Plan

ADR required: yes, for the child half only. Admitting a child's provider call under Capture On needs a ruling on who owns a child call's trace (owner, turn, segment); ADR-092 decision 3 says surviving fleet calls are captured, and ADR-337 owns calls by conversation, lineage, turn, run and call sequence, but nothing says which surface a child's request is fenced against.

1. Reproduce at the controller level (real controller, store, bridge, gateway and `ConsoleTraceBoundaryFactory`; HTTP-socket double in the OpenAI streaming shape) and in the real app (private child over the stock config, Full agent tools, Capture On default).
2. Separate the two requests: the parent's follow-up and the child's first request.
3. Parent follow-up: find why the probe saw `count_or_span`; correct the probe harness if it is an artifact.
4. Child: stop for the trace-ownership decision before any product change.

## Implementation Notes

Outcome: split. The parent's follow-up is a probe artifact (fixed in the harness); the child's request is a real refusal that needs a decision (AC #2 open).

**Parent's follow-up (probe artifact).** Not refused with a real response. Controller level: OpenAI streaming `tool_calls`, and at the `chat_api_call` layer four shapes (OpenAI `content: null` + `tool_calls` + `finish_reason`, the probe's `content: "finished"` + `tool_calls`, dict arguments, no call id) are all admitted. In the real app the follow-up was refused `count_or_span` (prefix 0, active 1, incoming 3) for `spawn_subagent` and for `calculator` alike, so it was never spawn-specific. Cause: not the response shape but the double's signature. The probe replaced `Chat_Functions.chat_api_call` with `def reply(**kwargs)`. The trace verifier projects final values through `project_chat_handler_kwargs`, which binds `inspect.signature(chat_api_call)`; with a bare `**kwargs` signature nothing binds, `messages_payload` is missing, and `verify_provider_request_shadow` returns `SANITIZER_FAILED`. The first call's surface is then one `provider_overlay` omission node, which no later tool-loop request can extend. With `functools.wraps(Chat_Functions.chat_api_call)` on the same double, both bundles verify and the follow-up reaches the provider.
- Harness fix: `Tests/Chat/test_console_default_send_capture.py` wraps `reply` and now asserts the captured surface has no omission node. Mutation: without the wrap, all three cases fail (`trace_omissions` 1 or 2).

**Child's request (real).** Refused in every run, controller level and real app, and the child run ends `error`; the parent finishes. It is not a surface-fence refusal:
- Usually `ValueError('trace_owner_unavailable')` (`Chat/console_trace_runtime.py`): `ConsoleAgentTraceRequestFactory.build` reuses the parent's admitted descriptors only for byte-equal rows, so the child's own system and task rows carry no saved revision, and the runtime finds no owner.
- When the task text equals a saved parent row (the model delegates the user's own words), that row binds the child to the parent's owner and turn: `TraceReservationRefused('trace_fresh_turn_requires_owned_recovery')` while the parent's call is open. So the child is then judged against the parent's turn and surface.
- Pinned by `Tests/Chat/test_console_spawn_subagent_trace_capture.py::test_child_request_after_native_spawn_reaches_provider` (strict xfail, both variants; `--runxfail` shows the two reasons above). `test_parent_follow_up_after_native_spawn_keeps_capture` passes: both parent calls complete, no refusal, no omission node.
- `test_parent_follow_up_after_native_spawn_keeps_capture` fails when the shadow projection is broken (the artifact's mechanism: `sanitizer_failed`, one omission node, follow-up refused).
- Decision needed before AC #2: who owns a child call's trace and which surface fences it. Options: (a) a child segment under the parent's owner and turn, fenced only against its own surface (ADR-092 rule 3 intact; ADR-337 amendment, largest); (b) children uncaptured with a gap event (smallest, but contradicts "uncaptured only by explicit choice", see `TraceReservationUnconfirmed`); (c) refuse `spawn_subagent` clearly while Capture On is in force (stop-gap). Recommended: (a), with (c) as the interim if a fix must ship first.
- Files: `Tests/Chat/test_console_spawn_subagent_trace_capture.py` (new), `Tests/Chat/test_console_default_send_capture.py`, `backlog/docs/lessons-testing-evidence.md`. No product code changed.
