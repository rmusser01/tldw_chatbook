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

**Owner decision 2026-11-17: option (a).** Sizing before code: (a) needs a ChaChaNotes schema step, so it was stopped for a split.
- Owner resolution is schema-level. Seven triggers (`console_trace_calls_owner_lineage`, `_events_owner_guard`, `_surface_nodes_owner_guard`, `_surface_replacements_owner_guard`, `_response_links_owner_guard`, `_segments_parent_owner_guard`, `_owners_active_prefix_guard`, all in `chachanotes_v55_to_v56_console_semantic_trace.sql`) and `ConsoleTraceRepository.get_effective_owner` find a segment's owner only by walking `parent_segment_id` up to an owner's `root_segment_id`. A segment with no link is refused at the first call insert (`provider call must use the active effective owner`).
- The existing fork link cannot carry a child: a segment with a parent must inherit the parent's surface up to `inherited_surface_head_id` (CHECK plus `console_trace_segments_inherited_surface`), so the child's first request would be judged against the parent's history again.
- GC keeps only segments reachable upward from attached or retained owner roots, plus segments with in-flight calls (`_mark_reachable_rows`). A child segment reachable from neither would lose its settled calls at the next GC.
- So (a) needs: a durable child-segment-to-owner link (new table or column, v86 to v87), the seven triggers recreated with that link, GC roots, `get_effective_owner`, then the runtime work (13 `owner.root_segment_id` sites in `console_trace_runtime.py`, a child marker from the factory). Split proposal in the stream report.

**Owner decisions 2026-11-17 (build A then B):** a new table, not a column; forks do not carry pre-fork child calls; an unresolved child call does not block the parent's recovery or fresh send; no interim refusal; trace UI for child calls is a later follow-up.

**PR A, store (branch `fu/41310a-trace-child-segments`), schema v86 to v87:**
- `console_trace_child_segments(segment_id PK, owner_id, turn_id, run_id UNIQUE)` links one root-less segment per child run to the parent's owner and turn. The run id is the child's provider-call chain (runtime state). Guards: insert (attached owner; empty, root-less segment that is no owner root or fork parent), append-only, GC-authorized delete.
- The five owner guards that resolve a segment's owner (`console_trace_calls_owner_lineage`, `_events_owner_guard`, `_surface_nodes_owner_guard`, `_surface_replacements_owner_guard`, `_response_links_owner_guard`) are recreated with one change: the ancestry starts at the linked owner's root when the segment is a child's. `ConsoleTraceRepository.get_effective_owner` does the same; `create_child_segment` / `get_child_segment(run_id)` added. The two fork-lineage guards are unchanged: a child segment is never a fork parent.
- GC roots include child segments of attached or retained owners; links are marked with their segment, swept with it, and their owner stays marked. Purge (detach) collects them.
- No index: the GC root join scans links and probes owners by primary key; the UNIQUE run id lookup uses its autoindex.
- Recovery: v87 catalog captured from a fresh constructor (`CHACHANOTES_V87_*` in `recovery_core_schema.py`), v86 kept as a migrating version, the restore sandbox runs the v87 step with an authorizer allowing exactly its objects, and the shared Subscriptions catalog gains the v87 lineage.
- Tests: `Tests/DB/test_chachanotes_v87_console_trace_child_segments.py` (child owned by the parent owner and holding its own calls; unlinked or foreign-owner calls refused by repository and trigger; link needs an attached owner and an empty root-less segment; GC keeps while attached and collects after detach; v86 upgrade keeps the trace and recreates the guards). Census pins updated: trigger digests, table columns, GC sweep targets and their evidence fingerprint, recovery policy, the prompt mix-in restore's target version.
- Mutations (each fails the new file): GC roots without the child branch (1 test), the calls guard with its original anchor (3), `get_effective_owner` without the child anchor (3).
