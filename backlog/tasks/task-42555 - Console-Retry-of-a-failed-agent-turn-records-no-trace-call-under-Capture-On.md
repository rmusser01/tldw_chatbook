---
id: TASK-42555
title: "Console Retry of a failed agent turn records no trace call under Capture On"
status: In Progress
assignee:
- '@claude'
created_date: 2026-11-26 04:50
updated_date: 2026-11-26 07:05
labels:
- follow-up
- console
- trace
priority: medium
references:
- tldw_chatbook/Chat/console_trace_service.py
- backlog/decisions/337-console-reference-backed-semantic-trace-ledger.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the TASK-42525 round-3 re-review (2026-11-25), outside that change's scope. With Capture On, the reviewer's probe of Console's Retry on a failed agent turn recorded no trace call at all, with or without a provider switch. The trace route registry lists RETRY as `CONVERSATION_TRACE`, so a retried agent turn is expected to be traced like any other captured call.

Either Retry reaches the provider without a trace call (a capture gap: a captured chat would send a request the trace never recorded), or Retry of a failed agent turn does not send at all (a broken action). Which one is not yet known.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The notes say what Retry of a failed agent turn does under Capture On today (whether the provider is called, what the trace records, what the user sees), shown by a test through the real controller with only the provider handlers faked
- [ ] #2 With Capture On, Retry of a failed agent turn either sends and is traced as a call of that turn, or is refused before any provider call with copy that says why; it never reaches the provider untraced
- [ ] #3 The test is red before the change and green after, and runs on a PR lane
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Probe through `_project_console` (real controller, bridge, gateway, trace factory; only `chat_api_call` faked): Capture On, an agent run whose calls all fail, then `retry_message`. Record provider calls, trace rows and what the user sees on dev 6e41919880.
2. Overlap check: run the same test on PR #3717's head (4bad6ab154) in a detached scratch worktree; remove it after.
3. If #3717 does not fix it: give Retry the Capture-On request a send gets (a turn preparation carrying the admitted policy, `_build_durable_trace_request`, route RETRY or DIRECT_PREFILL), so the Retry is reserved and recorded under its turn. Fail closed: a request that cannot be built, a temporary chat, or a trace refusal before dispatch refuses the Retry with copy, before any provider call. Keep the controller diff to the Retry body (#3717 edits other parts of the file).
4. Tests in a new file on a PR lane (recovered-suites): agent run traced (with and without AGENTS.md) and its next send; a run that answered a tool call first (refused before the provider: needs a rule ADR-337 does not grant, report it); the plain path traced and refused; Capture Off control; temporary chat. Red on dev, green after.
5. User Guide (semantic-trace-capture.md), size ratchet, lane timing, the brief's checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Status: decided, waiting on #3717 (2026-11-26).** Investigated; no product change committed. Tracing Retry works mechanically, but the next message after it cannot be witnessed in several common shapes without a rule ADR-337 did not grant. The owner chose the rule (below).

**Owner decision (2026-11-26), "Trace as a new run (Recommended)".** The question was: "With Capture On, what should Retry of a failed agent turn do?" The chosen option, verbatim: "New ADR-337 amendment: the Retry is traced as a new run of the same turn. The next message's checks count the turn's failed runs plus the Retry run as one set. A Retry's first call may drop the failed run's own tool rows, bounded like the fallback rewrite. Built after #3717 merges, reusing its retry record (TASK-33621.181)."

**Depends on:** PR #3717 (branch `fix/task-33621.6-retry-capture-regression`, still open and conflicting on 2026-11-26) and TASK-33621.181 (an explicit record of a Retry re-dispatch). Build this task only after #3717 merges, on top of that record. The ADR-337 amendment is written with the build, not before it.

**What Retry does today** (dev 6e41919880; real `ConsoleChatController`, `ConsoleAgentBridge`, `ConsoleProviderGateway` and `ConsoleTraceBoundaryFactory` through `Tests/Chat/test_console_project_instruction_traces.py::_project_console`; only `chat_api_call` faked by `_hosted_replies`). Capture On (effective, checked before Retry). An agent run whose three calls fail with 503 (`agent_first` error, `tool_loop` error twice), then **Retry** (`retry_message`):
- The provider is called once more (4 calls).
- The trace records nothing new: the three failed calls stay, and no call exists for the Retry in that turn or any other.
- The user sees an ordinary success: the run state is `completed` ("Response complete."), and the failed reply becomes `complete` with the provider's text.

So it is a capture gap, not a broken action. The plain-provider path (agent runtime off) does the same: the provider answers and no `retry` call is recorded.

**Cause.** A Retry has no turn preparation. `_retry_message_body` calls `_stream_assistant_response` with neither `preparation_id` nor `capture_mode_override`, and both `console_agent_reply._run_agent_reply` and `_run_direct_provider_reply` then fall back to `CAPTURE_OFF`. ADR-337's 2026-10-27 amendment records the same mechanism for Edit & resend and Regenerate ("it has no turn preparation"), which stay uncaptured by that amendment.

**#3717 (head 4bad6ab154) does not fix it.** The test was run in a detached scratch worktree at that head (helpers inlined, since that base predates them), then the worktree was removed. It failed exactly as on dev: 6 failed, 1 passed (the Capture Off control), with the same assertions (`assert [] == [('agent_first', 'complete')]`; the Retry sent and nothing traced). #3717 changes the trace card's Retry. Its own TASK-33621.171 records the owner's "record the retry" decision for Retry anyway as not implemented.

**Prototype** (not committed; its patch sat in a /tmp scratch dir that is cleared after about 3 days, so its design is recorded here).
- *Hook.* `_retry_message_body` (`Chat/console_chat_controller.py`) snapshots `saved_messages = tuple(dict(row) for row in provider_messages)` right after `_provider_messages_for_session(..., annotate_ids=True)`, before any transform. It then calls a new `stream_retry(controller, saved_messages, **stream_kwargs)` (in `Chat/console_turn_resend.py`) in place of `_stream_assistant_response`. That is 3 lines in the controller.
- *Capture decision.* The decision mirrors a send: Capture On when `capture_policy_snapshot(session).effective_capture_enabled` holds and the gateway has `supports_durable_capture`. Otherwise the Retry streams as before (Capture Off). An unreadable policy refuses the Retry, fail closed. A temporary (`ephemeral`) chat with Capture On also refuses: ADR-337 decision 15 allows no durable call before Save.
- *Request builder (`_retry_trace_request`).*
  - The current user is the last USER row before the failed assistant.
  - A `ConsoleTurnPreparation` serves only as a policy carrier and is never registered: a fresh `preparation_id`, `attempt_id = turn_context.library_authority.attempt_id`, origin `manual`, `executed_draft` the user's content (attachment ids when it is empty), state `ACCEPTED`, `capture_mode` CAPTURE_ON, and PII settings from the snapshot.
  - The prompt mix-ins are composed into `provider_messages` first, with `compose_current_mixin_messages(..., user_message_id=mixin_anchor_id or user.id, attempt_id=...)`, as `_submit_draft_body` does before freezing. The reply path's own composition is then a no-op. The first version skipped this and sent mix-in turns without the mix-in body (`'input' == 'body A\n\ninput'`).
  - `controller._build_durable_trace_request(...)` is called with that preparation, `trace_source_messages=saved_messages`, `echoed_user_id=user.id`, `committed_user_id=user.persisted_message_id`, and route `DIRECT_PREFILL` when a prefill is pinned, else `RETRY`. For a pinned prefill the messages first go through `_provider_messages_with_prefill`. The request carries route RETRY even on the agent path: the bridge's `ConsoleAgentTraceRequestFactory` derives the AGENT_FIRST and TOOL_LOOP calls from it. The plain path checks that the request's route matches the call's, and a FRESH-routed request failed there with `TraceProvenanceAlignmentError`.
- *Dispatch.* `_stream_assistant_response(**stream, capture_mode_override=CAPTURE_ON, trace_request=request)`. The plain path re-raises the trace's refusal as `TraceCallPersistenceError`, which comes before the adapter is entered. It was caught and returned as a refusal ("Trace capture could not start, so Retry was not sent. Try again, or turn Capture off for this chat to retry without it."), and a later Retry still worked. The agent path settles a refusal inside the run ("Agent run failed: trace capture could not be saved, so this request was not sent.").
- *Tests written for it* (to rebuild later): the agent run traced, with and without AGENTS.md, and its next send; a run that answered a tool call first; the plain path traced, and refused then retried; a Capture Off control; a temporary chat. 6 of the 7 were red on dev; all 7 were green with the prototype.

Results:
- **No mix-ins, no project instructions, all calls failed before any response:** the Retry is traced as a new first call of the same turn: `agent_first` complete on the agent path, `retry` complete on the plain path. It is response-linked, and the failed calls are unchanged. The next message is sent and traced.
- **With AGENTS.md:** the Retry is traced, but the next message stops at "Trace capture could not start" (`completed_tool_turn_lineage`). It is blocked on dev too, after the untraced Retry (`completed_tool_turn_unavailable`), and also with #3717 plus the prototype.
- **With prompt mix-ins** (two probes with one mix-in enabled, run after the composition fix): the Retry is traced, but the next message is blocked: `completed_tool_turn_lineage` on the agent path, `completed_tool_turn_unavailable` on the plain path (with or without a Regenerate in between). On dev it is sent. This holds with #3717 applied too.
- **First prototype, before the mix-in composition fix:** the covering run had 16 reds that are green on dev. Their causes were not separated, and at least the mix-in files also lost the mix-in body on the Retry's request. Two of them may not involve mix-ins at all (the controller and response-rules nodes):
  - `test_console_chat_controller.py::test_controller_retry_reuses_exact_owned_compound_reservation[retry-twice-agent]`;
  - `test_console_prompt_mixin_forks.py::test_durable_fork_immediate_publication_and_original_retry[False|True]`;
  - `test_console_prompt_mixin_journey.py::test_local_queue_retry_current_regenerate_reopen[False|True]`;
  - in `test_console_prompt_mixin_retry.py`: `test_continue_replay_composes_only_synthetic_current_user[retry-False|retry-True]`, `test_failed_turn_resend_saves_original_mixins_during_a_competing_write`, `test_frozen_replay_cannot_roll_current_accepted_layout_back[retry]`, `test_original_retry_ignores_unavailable_current_usage_preferences`, `test_original_retry_survives_reopen_deleted_source_and_latest_loss[False-True|True-True]`, `test_promoted_history_gets_no_seal_and_first_durable_trace_can_retry_off` and `test_retry_original_regenerate_current_and_clean_history[False|True]`;
  - `test_response_rules_runtime.py::test_real_agent_remaining_budget_limits_native_repairs[2-True]`.

- **After the composition fix,** only `test_console_prompt_mixin_retry.py` was rerun (with the 7 new tests): 3 failed, 104 passed. The other 13 nodes above were not rerun. The 3 reds:
  - `test_retry_original_regenerate_current_and_clean_history[True]`: the next send is blocked with `completed_tool_turn_unavailable`. The mix-in proof `ConsoleTraceService.operational_replacement_attempt` walks from the response's attempt back through `replaces_attempt_id` to an attempt whose `trace_call_id` is the origin call. The Retry's own attempt is never bound: `bind_attempt_trace_call` binds only through a native dispatch checkpoint, and a Retry has none. Its predecessor is bound to the failed FRESH call, not to the Retry's call.
  - `test_continue_replay_composes_only_synthetic_current_user[retry-False|retry-True]`: a Retry of a failed Continue reply. Its current turn is the synthetic `continue:<attempt>` anchor, which is not a saved revision, so the trace refuses it. Dev sends it untraced, like Continue. The build must decide this case: send it Capture Off like Continue, or refuse it.
- **A run that answered a tool call before failing:** refused at reservation (`unsupported_surface_change`) before any provider call. The run says "trace capture could not be saved, so this request was not sent". The Retry starts again from the user message, so its request drops the run's tool rows, a removal the owner's decision now allows (bounded like the fallback rewrite).

**Why a rule is needed.** The completed-turn witness and the mix-in and project source proofs follow one run per turn. A Retry answers in a later run of the same turn, while the turn's own rows (its user row, mix-in transform and project rows) were appended under the earlier, failed run. ADR-337 grants a run set only for provider fallback chains (TASK-42525). #3717's `_redispatched_head_call` infers a narrower case for the trace card's Retry (TASK-33621.181 asks to record it explicitly instead).

**Question put to the owner** (decided above). Under Capture On, how should Retry of a failed agent turn behave?
- (a) Traced as a new run of the same turn. This needs an ADR-337 amendment: the completed-turn witness, and the mix-in and project source proofs, treat a turn's failed runs plus its answering Retry run as one run set. A Retry's first call may also remove the failed run's own earned tool rows (bounded like TASK-42525's rewrite).
- (b) Traced as a continuation of the failed run: an unchanged in-place retry of its last failed call, reusing the run's actor, chain and frozen policy. This reopens a closed run, and a tool-history run would resume from its tool result instead of starting over.
- (c) Refused under Capture On ("turn Capture off to retry"). Simplest, but Retry is then unusable under the default Capture On.
- (d) Kept Capture Off with a visible, persisted "sent without capture" record (TASK-33621.171's AC#1 alternative). This needs AC#2 changed.

**Recommendation was (a)**, consistent with the owner's "record the retry" for TASK-33621.171 and the run set approved for TASK-42525. Implement it after #3717 lands, together with TASK-33621.181's explicit re-dispatch record, since all three touch the same witness code (`Chat/console_trace_continuation.py`, `Chat/console_trace_service.py`).

**Runs** (load 20-52): the prototype's 7 tests passed with the prototype and 6 failed on dev. The covering files (25, every test file that calls `retry_message`/`resend_turn`, plus the route census) ran at `-n 4` with the first prototype: 52 failed and 1121 passed. 36 of the 52 fail identically on dev; the other 16 are listed above. After the composition fix: `test_console_prompt_mixin_retry.py` plus the 7 new tests, 3 failed and 104 passed.
<!-- SECTION:NOTES:END -->
