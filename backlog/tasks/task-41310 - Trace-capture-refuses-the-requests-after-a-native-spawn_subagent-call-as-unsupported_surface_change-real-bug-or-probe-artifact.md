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
- [x] #2 If they are refused, both requests reach the provider with trace capture intact, pinned by a regression test through the real controller, store and trace runtime that is red on dev
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
- Size ratchets (serial `Tests/Architecture`): `ChaChaNotes_DB.py` may not grow, so the v86->v87 and v85->v86 steps share `DB/chachanotes_sql_steps.py` (22552 -> 22544); `console_trace_repository.py` crosses 4000 and gets a row at 4017 (B raises it to 4021).
- Commits: A = `942370e268`, `a12ca75c65`, `a9e1e2ed12` (the shared SQL step helper hid each step's entry-version guard from `test_every_migration_step_has_an_entry_version_guard`; the guard call is back in each step); B = `f95ed834aa`, `886bbecedf` (gateway size budget), rebuilt on A's tip. Verified 2026-11-17: A and B test files green except failures also red on base (ChaChaNotesDB bootstrap/trigger-census, test_core_owners schema policy); `Tests/Architecture` on B fails only persistent-diagnostic-inventory and timer-path inventory tests, in files neither branch touches. A trigger mutation fails 3 tests; B runtime revert fails 4.

**PR B, runtime (branch `fu/41310b-trace-child-runtime`, on top of A):**
- `ConsoleAgentTraceRequestFactory.build(child=True)` (passed by the streaming adapter for a sub-agent) reuses none of the parent's saved references and puts the spawning turn's saved user revision in the route provenance (`RequestRouteTraceProvenance.parent_revision_id`). The turn is read from the admitted request the parent run was built from; the gateway's dispatch check compares route, actor and chain only.
- `ConsoleTraceBoundaryFactory`: a child call takes owner and turn from that revision, its segment from `get_child_segment(run_id)` (created on its first call), and threads that segment through the surface fence, tool-loop chain checks, reservation, call-boundary event and boundary identity. It skips the parent's fresh-turn unresolved check and the mix-in attempt binding. `has_unresolved_call_for_turn` ignores child segments.
- ADR-337 and ADR-092 amended; `Docs/Architecture/database-layer.md` and `backup-recovery.md` describe v87.
- Tests: `test_child_request_after_native_spawn_reaches_provider[own-task|task-equals-user-message|child-tool-loop]` (was the strict xfail; now red on dev with the two refusal reasons, green here: the child reaches the provider, its calls are complete in its own linked segment under the parent's owner and turn, the parent's calls stay on the root); `test_parent_surface_still_refuses_a_changed_history_beside_a_child` (the parent's changed follow-up is refused `count_or_span` while the child is traced); `test_an_unresolved_child_call_never_blocks_the_parents_turn`.
- Mutations: reusing the parent's references (1 fails), no child turn (4), child on the root segment (5), the fresh-turn check applied to children (4), unscoped unresolved check (1), strict gateway route equality (4). Not covered: skipping the mix-in binding for a child (no test drives a mix-in turn with a child).
- Found, not fixed: a child's own native tool round on a provider-continuation endpoint fails before any trace ("Provider continuation could not be persisted", Capture Off too), so the child tool-loop case uses a fenced call. A child call settles with its parent's turn (the run's settlement sink), so a child call outliving a blocked parent turn stays `response_started`.

**Review round 2026-11-17 (A "approve with minor fixes", B "approve"); all findings fixed.**
- A (`806f97009e`): the calls guard now also requires the call's turn and run to equal the link's (`turn_id`/`run_id` are runtime identities with no table, so they cannot be tied to the owner's conversation in SQL; the migration header states this). `console_trace_owners_empty_root_guard` and `console_trace_segments_parent_owner_guard` refuse a linked child segment as an owner root or fork parent (seven guards recreated; recovery catalog, restore-sandbox allowance and trigger pins updated; pins sorted). New tests: failing v86->v87 statement leaves v86 with no new objects, wrong entry version refused, fork does not carry pre-fork child calls, retention root keeps link and segment through GC, guard refusals. GC retention branch comment notes the non-key `owner_id` join (no index, so no plan pin). Mutations (each fails exactly its test): calls turn/run clause, parent-owner clause, empty-root clause, retention branch.
- Base-red (checked on the exact parent `8145ca4f6f`, detached copy, removed after): the 4 trigger-census failures, `test_core_owners::test_factory_discovery_and_schema_policy_are_runtime_independent`, `test_console_trace_graph_gc::test_reopen_has_no_trace_payload_in_any_durable_owner`, and 105 more `Tests/ChaChaNotesDB` failures (109 total on the parent). A adds only `test_bootstrap_at_version_then_replay_matches_fresh_bootstrap[86]`, the same undeclared-tables cause as [82]-[85].
- B: the adapter re-derives the spawning turn and refuses a dispatched child whose route names another (`ConsoleAgentTraceRequestFactory.verify_child_route`; the gateway keeps comparing route, actor and chain only, since threading a new parameter through its entry points would break its size pin). `ConsoleRequestProvenance` refuses a child route beside saved message rows. `_turn_revision_id` documents why the admitted request's last user row is the spawning turn. Tests: two siblings and re-spawn with equal task text (own segment each), a child is never offered `spawn_subagent` (nested children are structurally unsupported), drifted route refused, unit tests for both new rules; the DB file pins "a child's open call never blocks the parent's turn, the parent's own does". Mutations: dropping the adapter call and dropping the aggregate rule each fail their test. `test_explicit_prepared_request_constructor_rejects_missing_on_provenance` is red on base too.
