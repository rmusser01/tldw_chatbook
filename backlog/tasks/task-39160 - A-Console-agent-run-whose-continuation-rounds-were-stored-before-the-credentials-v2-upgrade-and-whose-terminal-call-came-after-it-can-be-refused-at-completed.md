---
id: TASK-39160
title: "A Console agent run whose continuation rounds were stored before the credentials-v2 upgrade and whose terminal call came after it can be refused at completed-replay attachment"
status: Done
assignee: ['@claude']
created_date: 2026-11-08 06:54
updated_date: 2026-11-10 18:21
labels:
- follow-up
- trace
- console
priority: low
references:
- tldw_chatbook/Chat/console_trace_service.py
- tldw_chatbook/Chat/console_trace_continuation.py
- tldw_chatbook/Chat/console_trace_redaction.py
- Tests/Chat/test_console_dispatch_continuation_handoff.py
- Tests/Chat/test_console_completed_replay.py
dependencies:
- TASK-39123
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 12 of the 2026-10-23 burn-down (review r12b-rev3 of TASK-39123, round-12 branch fu/r12-backup). Read on the round-12 stack 9091820bf2, 2026-11-08; not built or reproduced. This code exists only in round 12, so it is named by symbol.

**What round 12 changed.** TASK-39123 stores new traces under `credentials-v2`. `trace_policies_agree` lets a `credentials-v1` round and a `credentials-v2` round agree.

**How the completed replay compares.**
- `ConsoleTraceService._completed_replay_trace_checkpoint(value, policy, …)` sanitizes the whole checkpoint under one version. That is the version of the policy it is given (`previous_policy`, the terminal call's), or the default when that version is not readable.
- Its caller in `console_trace_continuation.py` then compares each prior continuation node, as stored, with that checkpoint. If `prior.rounds != complete.rounds[: len(prior.rounds)]`, it raises `completed_replay_prior_attachment`.

**The case.** Take an agent run interrupted before the upgrade and resumed after it. Its early continuation rounds were stored under v1, and its terminal call is under v2. If an early round holds a bare `sk_live_`-style key, the replay compares v1-stored bytes with a v2-masked checkpoint, and the run would be refused. TASK-39123's own tests cover a v1 terminal followed by a v2 follow-up, not mixed versions inside one run's rounds.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A test builds a run whose early continuation rounds are stored under `credentials-v1` with a bare key and whose terminal call is `credentials-v2`, and shows whether the completed replay is refused; if it is, the test is red before the change (N/A — no live path records such a run: one policy per tool chain, console_trace_runtime.py:526-546 and :575-592; call policy_id immutable, chachanotes_v55_to_v56_console_semantic_trace.sql:977; a resumed run is Capture Off, console_agent_reply.py:350-354)
- [x] #2 Such a run's replay is admitted, with every captured row equal to what the current version masks; if it must stay refused, the notes say why and what the user sees (N/A — no run of that shape can exist to admit or refuse; see AC #1 and the trace in the notes)
- [x] #3 A replay whose rounds really changed is still refused (N/A — no code changed; changed rounds stay refused, pinned by test_console_dispatch_continuation_handoff.py::test_completed_replay_refuses_changed_history_before_dispatch[recorded_round], green 2026-11-10)
- [x] #4 The test runs on a PR lane (never the fast PR contract), with three timings recorded (N/A — no test was added: the case is not reachable)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace where a run's trace policy is frozen and whether a run's calls can carry different credential versions.
2. If such a run can be recorded, build it in `Tests/Chat/test_console_dispatch_continuation_handoff.py` (the upgraded-from-v1 harness) and show whether the completed replay refuses it; otherwise stop and report NEEDS_CONTEXT with the evidence.
3. (Coordinator ruling 2026-11-10) Trace the dispatch-recovery Retry and interrupted-run resume paths, and any other live writer; if none reaches the case, close it as not reachable with N/A ticks and the trace, and add no hardening code.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Closed as not reachable (coordinator ruling 2026-11-10: trace option 2 first; close if no live path reaches it). Traced on base ae9f9e8d21 (branch fu/r13-cprod); no code changed.**

Question: can any live path record a run whose continuation rounds are stored under `credentials-v1` and whose terminal call is `credentials-v2`, or put v1-stored nodes inside a completed tool turn's replay range under a v2 terminal? Every path checked:

1. **The policy is frozen per prepared turn.** `FrozenTracePolicy(..., credential_filter_version=CREDENTIAL_FILTER_VERSION)` is built once per turn in `_build_durable_trace_request` (`Chat/console_chat_controller.py:11994`), or per speculative voice attempt (`:9140`). Each provider call reads it from its request's provenance (`Chat/console_trace_runtime.py:440`).
2. **A tool chain keeps its origin's policy.** A TOOL_LOOP call is refused (`trace_tool_chain_unavailable`) unless the run origin's `policy_id` equals the call's (`console_trace_runtime.py:526-546`). It is also refused unless the previous call has the same `policy_id` and owns the surface tail (`previous.policy_id != policy.policy_id`, `previous.surface_node_id != tail.node_id`, `:575-592`). So every call of one run, the terminal included, carries one policy.
3. **A run cannot be re-entered with a new policy.**
   - An AGENT_FIRST call on an already-started run is refused (`trace_primary_run_already_started`, `console_trace_runtime.py:509`).
   - The only reuse of a reservation is the owned recovery, which requires `policy.policy_id == reserved.policy_id` (`:178-189`).
4. **A call's policy is immutable.** The `console_trace_calls_immutable_guard` trigger (`DB/migrations/chachanotes_v55_to_v56_console_semantic_trace.sql:977`) aborts any `UPDATE` of `console_trace_calls.policy_id`.
5. **An interrupted run resumed after the upgrade records no trace calls.** `recover_provider_continuation("resume" | "take_over")` goes to `_resume_provider_continuation`. That is the only caller that restores an interrupted run's rounds (`restore_provider_continuation=checkpoint`, `console_chat_controller.py:7135-7151`), and it passes neither a preparation id nor a capture override. `_run_agent_reply` therefore runs it Capture Off (`Chat/console_agent_reply.py:350-354`), and a Capture Off bridge carries no trace provenance (`Chat/console_agent_bridge.py:3349-3358`). No v2 terminal call is written for that run.
6. **A dispatch-recovery Retry replays the accepted turn as a new run.**
   - Every other `_run_agent_reply` caller (for example `console_chat_controller.py:25401`) passes no restored rounds.
   - So the Retry's request carries no rounds from the interrupted run. Its calls get one fresh policy (points 1 and 2).
   - Its TOOL_LOOP calls append on its own surface tail (point 2), so none of the interrupted run's continuation nodes sit on its chain.
7. **The replay range holds only the run's own nodes.**
   - The completed-replay check reads `plan.start_sequence..plan.end_sequence` (`Chat/console_trace_continuation.py:952`).
   - That range starts after the origin call's surface head (`witness.matches_start_sequence`, `Chat/console_trace_final_values.py:361-367`; checked at `console_trace_continuation.py:777`).
   - Every node in it that is not a replay continuation must be a `tool_call`/`tool_result` (or the restored source's `active_request`/`prompt_mixins`, or `project_instruction`; `:1000-1015`).
   - The replay continuation nodes it compares (`:955-997`) are therefore the run's own continuation deltas, appended under the run's single policy (point 2).
   - Cross-version prefix reuse (`ConsoleTraceService._durable_reference_matches`, `Chat/console_trace_service.py:5302-5386`) matches only nodes already on the call's surface chain, and inside the range those are the run's own.
   - The terminal's policy (`console_trace_continuation.py:752`) is the policy those nodes were stored under.
   - A forked segment reads the same original nodes through `read_lineage_surface_nodes`, so it does not change this.
8. **Legacy snapshot import** (`Chat/console_trace_legacy.py:255-264`) writes `route_identity="legacy_snapshot"` calls (`:293`, `:388`), never an AGENT_FIRST/TOOL_LOOP chain, so it never forms a completed tool turn.

**Evidence that behaviour is unchanged** (`-n 4`, load ~48): `test_completed_replay_refuses_changed_history_before_dispatch[recorded_round]` and `test_completed_replay_continues_a_turn_traced_under_credentials_v1[safe|full]`, 3 passed.

**Found while tracing (not this task; not reproduced):** the legacy import calls `ensure_policy` with the fixed `_LEGACY_POLICY_ID` (`console_trace_legacy.py:44`) and the current version. `ensure_policy` raises `TraceIdentityConflict("policy_id")` when a stored row differs (`Chat/console_trace_repository.py:2365-2368`). So a profile that ran a legacy import before the v2 upgrade may refuse every later legacy import. Proposed as a follow-up.
<!-- SECTION:NOTES:END -->
