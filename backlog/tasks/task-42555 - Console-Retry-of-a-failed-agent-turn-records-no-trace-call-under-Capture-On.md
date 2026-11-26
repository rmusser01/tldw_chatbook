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
**Status: NEEDS_CONTEXT (owner decision), 2026-11-26.** Investigated; no product change committed. Tracing Retry works mechanically, but the next message after it cannot be witnessed in several common shapes without a rule ADR-337 does not grant. The prototype and its tests are kept outside the branch (see "Prototype" below).

**What Retry does today** (dev 6e41919880; real `ConsoleChatController`, `ConsoleAgentBridge`, `ConsoleProviderGateway` and `ConsoleTraceBoundaryFactory` through `Tests/Chat/test_console_project_instruction_traces.py::_project_console`; only `chat_api_call` faked by `_hosted_replies`). Capture On (effective, checked before Retry). An agent run whose three calls fail with 503 (`agent_first` error, `tool_loop` error twice), then **Retry** (`retry_message`):
- The provider is called once more (4 calls).
- The trace records nothing new: the three failed calls stay, and no call exists for the Retry in that turn or any other.
- The user sees an ordinary success: the run state is `completed` ("Response complete."), and the failed reply becomes `complete` with the provider's text.

So it is a capture gap, not a broken action. The plain-provider path (agent runtime off) does the same: the provider answers and no `retry` call is recorded.

**Cause.** A Retry has no turn preparation. `_retry_message_body` calls `_stream_assistant_response` with neither `preparation_id` nor `capture_mode_override`, and both `console_agent_reply._run_agent_reply` and `_run_direct_provider_reply` then fall back to `CAPTURE_OFF`. ADR-337's 2026-10-27 amendment records the same mechanism for Edit & resend and Regenerate ("it has no turn preparation"), which stay uncaptured by that amendment.

**#3717 (head 4bad6ab154) does not fix it.** The test was run in a detached scratch worktree at that head (helpers inlined, since that base predates them), then the worktree was removed. It failed exactly as on dev: 6 failed, 1 passed (the Capture Off control), with the same assertions (`assert [] == [('agent_first', 'complete')]`; the Retry sent and nothing traced). #3717 changes the trace card's Retry. Its own TASK-33621.171 records the owner's "record the retry" decision for Retry anyway as not implemented.

**Prototype** (`$FU/r27rt-42555/`: `prototype-src.patch`, `test_console_retry_failed_agent_turn_trace.py`, two probes). Under Capture On, `stream_retry` (in `Chat/console_turn_resend.py`) gives the Retry the admitted request a send gets:
- a `ConsoleTurnPreparation` carrying the policy;
- the mix-ins composed first;
- `_build_durable_trace_request`, with route RETRY, or DIRECT_PREFILL for a pinned prefill.

It fails closed with copy for an unbuildable request, a temporary chat, or the plain path's trace refusal. Results:
- **No mix-ins, no project instructions, all calls failed before any response:** the Retry is traced as a new first call of the same turn (`agent_first` complete on the agent path, `retry` complete on the plain path), response-linked, with the failed calls unchanged. The next message is sent and traced. The 7 new tests were red on dev (6) and green with the prototype.
- **With AGENTS.md:** the Retry is traced, but the next message stops at "Trace capture could not start" (`completed_tool_turn_lineage`). It is blocked on dev too, after the untraced Retry (`completed_tool_turn_unavailable`), and also with #3717 plus the prototype.
- **With prompt mix-ins:** the Retry is traced, but the next message is blocked: `completed_tool_turn_lineage` on the agent path, `completed_tool_turn_unavailable` on the plain path. On dev it is sent. This holds with #3717 applied too. 16 existing tests turn red, all green on dev:
  - the mix-in retry, fork and journey files;
  - `test_controller_retry_reuses_exact_owned_compound_reservation[retry-twice-agent]`;
  - `test_real_agent_remaining_budget_limits_native_repairs[2-True]`.

  Not shippable.
- **A run that answered a tool call before failing:** refused at reservation (`unsupported_surface_change`) before any provider call; the run says "trace capture could not be saved, so this request was not sent". The Retry starts again from the user message, so its request drops the run's tool rows, a removal ADR-337 grants no call.

**Why a rule is needed.** The completed-turn witness and the mix-in and project source proofs follow one run per turn. A Retry answers in a later run of the same turn, while the turn's own rows (its user row, mix-in transform and project rows) were appended under the earlier, failed run. ADR-337 grants a run set only for provider fallback chains (TASK-42525). #3717's `_redispatched_head_call` infers a narrower case for the trace card's Retry (TASK-33621.181 asks to record it explicitly instead).

**Question for the owner.** Under Capture On, how should Retry of a failed agent turn behave?
- (a) Traced as a new run of the same turn. This needs an ADR-337 amendment: the completed-turn witness, and the mix-in and project source proofs, treat a turn's failed runs plus its answering Retry run as one run set. A Retry's first call may also remove the failed run's own earned tool rows (bounded like TASK-42525's rewrite).
- (b) Traced as a continuation of the failed run: an unchanged in-place retry of its last failed call, reusing the run's actor, chain and frozen policy. This reopens a closed run, and a tool-history run would resume from its tool result instead of starting over.
- (c) Refused under Capture On ("turn Capture off to retry"). Simplest, but Retry is then unusable under the default Capture On.
- (d) Kept Capture Off with a visible, persisted "sent without capture" record (TASK-33621.171's AC#1 alternative). This needs AC#2 changed.

**Recommendation: (a)**, consistent with the owner's "record the retry" for TASK-33621.171 and the run set approved for TASK-42525. Implement it after #3717 lands, together with TASK-33621.181's explicit re-dispatch record, since all three touch the same witness code (`Chat/console_trace_continuation.py`, `Chat/console_trace_service.py`).

**Runs** (load 20-52): the prototype's 7 tests passed with the prototype and 6 failed on dev. The covering files (25, every test file that calls `retry_message`/`resend_turn`, plus the route census) ran at `-n 4`: 52 failed and 1121 passed with the prototype. 36 of the 52 fail identically on dev; the other 16 are the mix-in regressions above.
<!-- SECTION:NOTES:END -->
