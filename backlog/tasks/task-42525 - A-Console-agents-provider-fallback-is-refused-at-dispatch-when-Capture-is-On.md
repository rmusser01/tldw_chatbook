---
id: TASK-42525
title: "A Console agent's provider fallback is refused at dispatch when Capture is On"
status: In Progress
assignee:
- '@claude'
created_date: 2026-11-23 23:00
updated_date: 2026-11-25 07:40
labels:
- follow-up
- console
- agents
- trace
priority: medium
references:
- tldw_chatbook/Agents/agent_runtime.py
- tldw_chatbook/Agents/agent_service.py
- tldw_chatbook/Chat/console_agent_bridge.py
- tldw_chatbook/Chat/console_trace_service.py
- Tests/Chat/test_console_failed_call_retry_ownership.py
- backlog/decisions/110-cross-provider-fallback-and-history-projection.md
- backlog/decisions/200-preset-pre-tool-fallback-targets.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the round-24 review of TASK-42472 (2026-11-23) and reproduced on fu/r24-diag (dev b31e3e9241 plus round 24). Paths are under `tldw_chatbook/` unless they start with `Tests/`.

**Why.** Capture is on by default for saved chats, and with Capture On a user's configured provider fallback never runs. When an agent's model call fails, `Agents/agent_runtime.py::_recover_model_failure` first retries in place. After the retries are spent (a transient error), or at once (a 402/403 credit-terminal error, or `ChatModelUnavailableError` for a preset target), it switches to the next ready fallback provider. Users set that chain in `[console] agent_fallback_providers` (`Chat/console_agent_bridge.py::console_fallback_providers`, wired into `AgentConfig.fallback_providers`) or with preset fallback targets. The switched call is the run's next call, so it goes out as a TOOL_LOOP call with a different provider, model and endpoint.

Since c24acf71683aa2f3f6a0a9c459f7194b284a7f3b (TASK-42472), llama.cpp direct HTTP failures are transient too (an `httpx.HTTPStatusError` with a retryable status, a timeout, a refused connection or a protocol error). Once a llama.cpp run's in-place retries run out, it now reaches the fallback as well, and hits the same refusal.

The previous call is in `error`, so `ConsoleTraceService.persist_request` (`Chat/console_trace_service.py`, the `previous.state is TraceCallState.ERROR` check) compares the two request headers. Provider, model and endpoint differ, so it raises `ValueError("trace_tool_chain_unavailable")`. This is the rule "a failed call authorizes an unchanged retry", pinned by `Tests/Chat/test_console_failed_call_retry_ownership.py::test_failed_call_retry_requires_exact_durable_chain[changed_provider|changed_model|changed_endpoint]`. The rule is right for a retry, but it also catches the fallback, which is meant to change the target.

**Evidence.** A probe, deleted after use, ran a real `ConsoleChatController` + `ConsoleAgentBridge` + `ConsoleProviderGateway` agent send:
- *Setup.* The real `ConsoleTraceBoundaryFactory` and Capture On. Only the provider handler was faked (`chat_api_call_fn`). `agent_fallback_providers = ["groq"]` was injected through a test-local `get_cli_setting` wrapper, and a fixture `GROQ_API_KEY` was set in the test process only. The primary (deepseek) always failed.
- *Results, 8 runs:*
  - `credit402`, complete and streaming: the fallback to groq was selected; `persist_request` raised `trace_tool_chain_unavailable`; the run ended `blocked` with blocker `trace_call` and "Trace capture could not start. Retry, Send without capture, or Cancel." The groq handler was never called. Lineage: `agent_first=error`, then a call left in `reserved`.
  - `rate429_exhausted`, complete and streaming: the two in-place retries were admitted (`tool_loop=error` twice). The fallback was then refused the same way.
  - The same four with Capture Off: all completed, and the groq handler answered.

**The two options.**
1. *A fallback starts a new trace chain.* The fallback call gets a fresh route identity (a new actor/chain, an AGENT_FIRST-like boundary for the new provider), recorded as linked to the failed chain. The retry rule stays as strict as it is.
2. *The retry check admits a target change marked as a fallback.* The gateway passes a typed fallback marker. `persist_request` then accepts a different provider, model, endpoint and generation settings for that one call, and records the switch in the request header.

**Recommendation: option 1.** A fallback can also rewrite the history. The branch runs `state.messages[:] = project_history_for_protocol(state.messages, native=candidate.native)`, because "the run now IS the new protocol". So the fallback request is not the same request sent to another provider. Option 2 would have to widen the failed-call check to admit surface changes as well, which is exactly what that check exists to refuse. A new chain records the fallback as the new provider conversation it is.

Also decide what happens to the call the refusal leaves in `reserved`.

**Owner decision (2026-11-24, recorded as "2 ok", approving the recommendation):** under Capture On, a provider fallback starts a **new trace chain** (option 1). The failed-call retry rule stays as strict as it is. Recorded in ADR-337 as the owner-approved amendment "a provider fallback starts a new chain (TASK-42525)".

**Second owner decision (2026-11-24, via AskUserQuestion: "Allow it, do it now (Recommended)"):** a fallback chain's first call may record one bounded rewrite of the recorded history. It may remove the leading system-prompt slot plus the run's own tool rows. Each removed row must pass the same run-ownership checks the next-send witness uses, within the 256-row limit. This closes AC#1 for the two shapes the first design left refused: a native/text tool-call re-encoding after tool activity, and llama.cpp falling back to a hosted provider. Recorded in ADR-337 as the owner-approved amendment "a fallback chain's first call may rewrite the run's tail (TASK-42525)".

**Third owner decision (2026-11-25, via AskUserQuestion: "Allow it, do it now (Recommended)"):** "A new turn's first call may put the leading system-prompt slot back when the recorded call had no system slot and the new request starts with a system prompt; it is the mirror of the removal you approved, recorded as a third ADR-337 amendment." Scope: only a new turn's first call (AGENT_FIRST or FRESH), never a retry, a fallback or a tool-loop call; only the leading slot, of kind rendered_system; only when the recorded surface has no leading system row (none at all, or removed by a fallback's marker) and the request's first row is a system prompt; every other row meets the existing rules unchanged; rechecked at dispatch binding and at the next send. A captured manual llama.cpp -> groq switch with AGENTS.md stays out of scope (a follow-up). Recorded in ADR-337 as the owner-approved amendment "a new turn's first call may put the leading system slot back (TASK-42525)".
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 With Capture On, an agent run whose primary provider fails switches to a ready configured fallback provider and completes, as it does with Capture Off
- [x] #2 The captured trace shows the failed calls and the fallback call, and which provider each went to
- [x] #3 A retry after a failed call that changes the provider, model or endpoint without being a fallback is still refused; the existing `changed_*` cases are still refused
- [x] #4 No call is left in `reserved` after a refused or completed fallback
- [x] #5 A test drives a Capture-On agent run through a fallback; it is red before the change and green after
- [ ] #6 With Capture On, after a llama.cpp run falls back to a hosted provider, the next message sent back to llama.cpp is sent and traced, with and without tool activity and with and without project instructions
- [ ] #7 With Capture On, a manual switch from a hosted provider to llama.cpp is sent and traced, with and without project instructions
- [ ] #8 Putting the system slot back on any call but a new turn's first, anywhere but first, as a second system row, over a slot the surface already has, or together with a change the existing rules refuse, is still refused before anything is reserved
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Record the owner decision here and write the new-chain design into ADR-337 as an owner-approved amendment before any code.
2. AC#5 first: add a Capture-On agent-run test (real controller, bridge, gateway and trace factory; only provider handlers faked) whose primary fails and whose configured fallback answers; show it red on the base (the run ends on the primary's error).
3. Remove the round-24 cut-off (`captured_fallback_providers`): a captured run plans the configured chain again; invert its pin in `Tests/Chat/test_rate_limit_reads_as_rate_limit.py`.
4. Bridge adapter (primary, Capture On): when a call names a target other than its chain's, require it to be the next unused entry of the run's configured fallback list, then mint a new chain id under the same actor. Refuse any other target change before preparation.
5. Trace factory: admit a TOOL_LOOP call at sequence 0 only as a fallback chain root linked to the failed call directly before it (same actor, other chain, settled ERROR with no response, same owner/segment/turn/policy, unchanged surface, nothing appended or replaced, provider or model changed). Resolve a run's origin through its fallback links for TOOL_LOOP admission and the completed-turn witness; generalise the witness's same-run lineage checks to the run's linked chains.
6. Tests: AC#1/#2/#4 (lineage and providers, no `reserved`), AC#3 (`changed_*` still refused, plus a same-target new chain), next send after a completed, a failed and an interrupted-then-discarded fallback run, and the hard negatives (unconfigured target, unearned tool history, a second fallback beyond the list, after a response-bearing call, cross-turn and cross-owner links), each with a mutation run.
7. User Guide (`semantic-trace-capture.md`, fallback docs if they mention Capture), size ratchets, lane timing, the brief's checks.

Second phase (owner decision 2026-11-24, "Allow it, do it now"; rebased onto trace1's restack `10a2fbd4ae`):
8. Write the bounded rewrite into ADR-337 as a second owner-approved amendment before any code: which rows, which checks, the bound.
9. Admit it only on a fallback chain's first call: remove the leading rendered-system slot (a removal-only replacement whose marker every projection drops) and replace the run's earned tool rows at the tail one for one; recheck at binding; let the completed-turn witness and `run_chains` follow a rewriting first call.
10. Tests: both shapes red -> green with their next sends; hard negatives each refused with mutations.
11. Move `Tests/Chat/test_console_failed_call_retry_ownership.py` from the Dreams riders step to dreams-boundary's admission-sensitive step (coordinator's budget decision), with both steps' budget comments and every lane contract.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Status:** AC#1-#5 are met. AC#1 was first met only in part (two shapes refused); the owner's second decision (2026-11-24) granted the rewrite that closes it (see "Second decision: the bounded rewrite").

**Decision.** Owner decision 2026-11-24 ("2 ok", approving the recommendation): under Capture On, a provider fallback starts a new trace chain. ADR-337 now carries it as the owner-approved amendment "a provider fallback starts a new chain (TASK-42525)", written before the code.

**What "new chain" means in the trace model.**
- The fallback keeps the run's actor and mints a new chain, so its calls are run `actor:new_chain`.
- The chain's first call is its sequence 0, on the TOOL_LOOP route. It is the run's next model call and carries the run's history.
- The link to the failed chain is the shared actor plus call-boundary order: the first call directly follows the failed call, or that call's failed llama.cpp resend. No schema change.

**Changes.**
- The round-24 cut-off `captured_fallback_providers` is deleted, so a captured run plans its configured chain again (`Chat/console_agent_bridge.py`, `Chat/console_agent_preparation.py`).
- The streaming adapter (`Chat/console_streaming_request.py`) handles a primary Capture-On run's calls. A call that names a target other than its chain's starts a new chain only when the target is the next unused entry of the run's configured fallback list. `_StreamingModelAdapter` now receives that list (`config.fallback_providers`). Any other switch raises before preparation, so nothing is reserved.
- The trace factory (`Chat/console_trace_runtime.py`) admits a TOOL_LOOP call at sequence 0 only as such a chain's first call. Its previous call is the linked failed call (`fallback_parent`). That call must be an ERROR with no response start, and the new call must go to another provider or model (`falls_back_from`). The existing TOOL_LOOP checks apply unchanged: latest boundary, policy, surface tail, and the origin's owner, segment, turn, policy and source pin. Like a retry, the new call may append and replace nothing. Its own rendered system row is not compared with the failed call's, and no header comparison applies (sequence 0 has no in-chain predecessor). The run's origin is found through the links (`run_chains`).
- The completed-turn witness (`Chat/console_trace_continuation.py::validate_completed_tool_turn`) treats a run as its AGENT_FIRST chain plus the fallback chains linked from it. Removed nodes may come from any of them: tool-loop calls at sequence 1 or later, each after a response-bearing call in its own chain (TASK-42534). Boundaries between origin and terminal may belong to any of them. Every link is rechecked from the ledger.
- **AC#4.** Every fallback refusal happens before a reservation commits: in the adapter, or in the reservation transaction, which rolls back. So no refused fallback leaves a `reserved` call. A completed fallback settles normally. The round-24 probe's leftover `reserved` call came from the refusal at dispatch binding, which no longer happens for a fallback.
- The User Guide is updated: `console.md` (the rate-limit paragraph said a captured chat never falls back) and `console/semantic-trace-capture.md`. The bridge's size row is lowered to its exact 10,120 (-22, dated).

**Evidence** (`Tests/Chat/test_console_failed_call_retry_ownership.py`, real controller, bridge, gateway and trace factory; only the provider handlers are faked).
- **Red/green.** With the branch's tests on the base's source (trace1 tip `95a687ba76`), 28 of the selected tests fail and 30 pass. Every new behaviour test is in the 28: the four fallback runs, the next-send matrix, the interrupted run, the adapter list cases, the factory's `valid` and `new_instructions` cases, the text-tool fallback, the link recheck and the re-encoding refusal. The renamed `test_a_captured_run_plans_its_configured_fallback_chain[CAPTURE_ON]` is red there too. The 30 that pass are refusals the base also makes, and pre-existing tests the selection matched. Before the cut-off was removed, the AC#5 test failed with the provider's own error. With only the cut-off removed, it failed with "Trace capture could not start" (`unsupported_surface_change` for the tool-history run).
- **Mutation runs** (each guard disabled alone; the tests it turned red):
  - adapter list membership: `unconfigured`, `beyond_the_list`, `back_up_the_list`;
  - adapter list position: `back_up_the_list`;
  - no new chain: 23 tests;
  - target change: `same_target` and the existing `changed_chain` cases;
  - no response start: `after_partial_response`;
  - ERROR state: the link recheck `[unsettled]`;
  - no append: `unearned_history`;
  - shared actor: `other_actor`;
  - the `run_chains` link recheck: both recheck cases;
  - rendered-system skip: `new_instructions`;
  - the witness's run set, bore-response run, foreign-boundary set and origin: 9, 3, 5 and 13 tests;
  - TOOL_LOOP origin through the links: 25 tests;
  - the factory's witness origin: 14 tests.

  `cross_turn` is refused by three independent checks: the origin's turn, its source pin, and no-append. Each alone leaves it refused; disabling all three admits it (red). `cross_owner` cannot be expressed: the link is looked up only in the new call's own owner segment, and its origin must hold the same owner. The test pins the refusal; no mutation of a single guard admits it.
- **Hard negatives.** Each is refused before anything is reserved, and the provider is never asked:
  - a chain that is not a configured fallback: the adapter cases, and `same_target` in the factory;
  - unearned tool history: `unearned_history`;
  - a second fallback beyond the list: `beyond_the_list`, `back_up_the_list`;
  - a fallback after a response-bearing call: `after_response`, `after_partial_response`;
  - cross-turn and cross-owner links: `cross_turn`, `cross_owner`;
  - a chain of another actor: `other_actor`.
- **AC#3.** All 40 `test_failed_call_retry_requires_exact_durable_chain` cases pass unchanged, every `changed_*` included. `changed_chain` (a new chain to the same target) is now refused by the target-change rule.
- **Next send.** It is sent and the run's calls are unchanged after:
  - completed fallback runs (`credit`, `tool_then_credit`, `credit_then_tool`);
  - failed ones (`credit_then_failed`, `tool_then_credit_then_failed`, `credit_then_tool_then_failed`), each with and without project context;
  - an interrupted-then-discarded fallback run (startup recovery, Discard, send).

  TASK-42534's rule holds across both chains: `test_a_fallback_run_never_lends_its_tool_history_a_failed_call` refuses the next message when the call that answered with the tool call, in either chain, reads as failed.
- **Flake found and fixed in a new test.** The first mutation round showed the interrupted test failing in about half the runs. `recover_open_calls` orders by `run_id`, and each chain's id is random, so the recovered list's order varied. The assertion now sorts.
- **Config reads.** RecoveryRequired recorder (`r24rrcount`): 0 over the 58 selected tests (35 new), and 0 over the llama.cpp limit pair. The file stays `bootstrap_profile`: the fallback builder reads `api_settings.<provider>.model`.
- **Runs.**
  - The touched files (`test_console_failed_call_retry_ownership.py`, `test_rate_limit_reads_as_rate_limit.py`) were soaked 20 rounds at `-n 4`: 182 passed in every round, at load 7.8-12.5.
  - Every `Tests/Chat/test_console_trace_*.py`, the touched files, the provider-gateway, fallback and bridge files, and 10 adapter callers ran at `-n 4`: 2768 passed, 53 failed. All 53 are known or red on the base's source:
    - TASK-42513's 32;
    - TASK-32032's `[stopped]`;
    - the 3 `test_console_provider_gateway.py` reds;
    - 17 in `test_console_continuation_group_merge.py`, `test_console_named_agent_model.py`, `test_console_redirect_stream_cut.py` and `test_console_request_provenance_routes.py`, red identically with the base's source.
  - `Tests/CI` at `-n 4`: 1946 passed, 2 skipped.
  - `Tests/Architecture` serially: 1507 passed, 5 skipped.
  - `./scripts/preflight.sh` passed. `ruff check tldw_chatbook --select F` is clean, and `Tests/lint/ruff_gate.py` reports RATCHET OK.
  - The Dreams tracking riders step (`r19-step.sh`, `-p no:xdist --timeout=180`): 1618 passed, 6 skipped in 1027.1 s, at load 10-12 with the soak running alongside. Rerun on the final tree (with the llama.cpp limit pair): 1620 passed, 6 skipped in 1079.1 s.
  - The llama.cpp limit pair and the text-tool fallback were soaked a further 20 rounds at `-n 2`: 3 passed in every round, at load 7.4-9.4. On the final tree: the touched files 184 passed at `-n 4`, `Tests/CI` 1946 passed, preflight passed.
- **Lane cost.** The file is already a whole-file target of the Dreams tracking riders step. The first 35 new tests take 73.8 s serially with the step's flags (`-p no:xdist --timeout=180`), at load 9 with other work running. The llama.cpp limit pair adds about 8 s (4.2 s and 4.1 s). Census and workflow are unchanged.

**Second decision: the bounded rewrite.** The first design left two shapes refused, both because the fallback's request had to replace part of the trace surface: a native/text tool-call re-encoding after tool activity, and a llama.cpp run (system prompt as the first message) falling back to a hosted provider (system prompt outside the message list). The owner allowed one bounded rewrite on a fallback chain's first call. It is written into ADR-337 as a second owner-approved amendment, before the code.
- **What it removes.** The leading rendered-system slot, only when it is the first active entry, the failed call's own header carried it, and the new request has no leading system row. The run's tool rows at the tail, replaced one for one, kind for kind, by the new call's rows.
- **Checks.** Every removed tool row must be earned: appended by a tool-loop call of one of the run's chains, in the run's owner, turn and policy, after a response-bearing predecessor in its chain, or written by an earlier fallback rewrite whose own removed rows pass that check. At most 256 rows. Rechecked in the dispatch-binding transaction, where only a reserved sequence-0 call linked to the failed call may carry it.
- **Recording.** The tool rows use an ordinary bounded replacement. The slot uses a removal-only replacement whose node is a content-free `surface_removal` marker that every projection drops (`TraceOmissionReason.SURFACE_SLOT_REMOVED`, no schema change). The removed rows' bytes and the earlier calls' heads are untouched.
- **Next send.** `run_chains` now follows a first call that rewrote the surface from the failed call's head, and the completed-turn witness accepts the rewritten rows: their sources, dead inside the range, are rechecked there under the TASK-42534 rule.
- **New refusal codes.** `fallback_rewrite_shape`, `fallback_rewrite_unavailable` and `trace_fallback_rewrite_unavailable` join the send-diagnostic categories (`Chat/console_send_diagnostics.py`). TASK-42471's census test failed until they did.
- **Files.** New `Chat/console_trace_fallback_rewrite.py` (planning, recheck, earned check, slot removal). `Chat/console_trace_final_values.py` (`FallbackRewriteWitness` and its shape), `Chat/console_trace_service.py` (threading, binding recheck, marker projection), `Chat/console_trace_runtime.py`, `Chat/console_trace_continuation.py`, `Chat/console_trace_provenance.py`. The trace service's size row rises by its exact +100 (dated).

**Second-phase evidence.**
- **Red/green.** Run on the pre-rewrite source (`88b33e8d4a`), the shape tests failed: `test_a_fallback_rewrites_the_runs_tool_calls_for_its_provider` (2) and `test_a_captured_llamacpp_run_falls_back_to_a_hosted_provider` (4) each read `'blocked' == 'completed'`. The low-level rewrite tests could not import the new module. All pass after.
- **Mutations** (each guard disabled; the tests turned red):
  - the bound: `over_256`;
  - the run set: `foreign_run`, earned `[run]`;
  - policy, owner, turn: earned `[policy]`, `[owner]`, `[turn]`;
  - an unanswered predecessor: earned `[unanswered]`;
  - the slot's run ownership: `slot_not_the_runs`;
  - the second-fallback branch: `valid_second_rewrite`;
  - the marker projection: all four llama.cpp shapes;
  - the witness's rewritten rows, dead rows, and `run_chains` following a rewrite: shape (a) with project context;
  - the factory passing a rewrite to an in-place retry: `retry_rewrite`;
  - the factory passing one to the chain's second call: `second_call`;
  - the binding-time first-call check alone: nothing, because the factory refuses first. With the factory mutation, the refusal moves to binding and leaves a reserved call (red on AC#4). With both, the rewrite is admitted (red).
  - `user_row` is refused by three checks: row kind, the appending call's route, and the sequence-0 branch's need for a recorded rewrite. Each alone leaves it refused; all three disabled admit it (red).
- **Next sends.** After shape (a) the next message is sent, with and without project context. After shape (b) it goes back to llama.cpp and is refused (`unsupported_surface_change`); Send without capture then sends it. This is not new: a probe found the same refusal for an ordinary captured switch from a hosted provider to llama.cpp, with and without project context. Putting a leading system row back was not part of the decision.

**Second-phase runs** (tree after the rebase onto `10a2fbd4ae` and the rewrite):
- **Soak.** The touched test files (the retry file, the rate-limit file and the trace diagnostic census) ran 20 rounds at `-n 4`: 220 passed in every round, at load 8.3-15.9.
- **Wide run.** Every `Tests/Chat/test_console_trace_*.py`, the touched files, the provider-gateway, fallback and bridge files, and the adapter callers ran at `-n 4`: 2834 passed, 22 failed. TASK-42513's 32 are green now, after reds2. The 22 failures are:
  - TASK-32032's `[stopped]`;
  - the 3 gateway reds;
  - the 17 base-identical reds listed below;
  - the diagnostic census. That one was mine (the new refusal codes), and it is fixed above.
- **Other checks.**
  - `Tests/CI` at `-n 4`: 2009 passed, 2 skipped.
  - `Tests/Architecture` serially: 1507 passed, 5 skipped.
  - `./scripts/preflight.sh` passed. `ruff check tldw_chatbook --select F` is clean, and `ruff_gate` reports RATCHET OK after one B009 fix in the new module.

**Lane move (coordinator's decision).** `Tests/Chat/test_console_failed_call_retry_ownership.py` moved from the Dreams tracking riders step to dreams-boundary's "Run Dreams admission-sensitive contracts". That step runs only `bootstrap_profile` files, as this file is. The move updated `DREAMS_TRACKING_RIDERS`, `DREAMS_ADMISSION_TARGETS` and `test_response_rules_critical_gate.py`'s pinned argv, and the file has one owner. The whole file is 166 tests in 250.4 s serially with the step's flags (`-p no:xdist --timeout=300`) at load ~6. Both steps after the move, serially via `r19-step.sh`: the admission-sensitive step, 6 targets, 276 passed in 370.4 s at load 5-7; the riders step, 42 targets, 1595 passed and 6 skipped in 759.3 s at load 5-6.

**Limits and finds (proposed follow-ups).**
- Putting a leading system row back on a captured send is refused: after a llama.cpp fallback to a hosted provider, and on any captured hosted -> llama.cpp switch. That needs an owner (ADR) choice; recommendation below in the report.
- A spawned child's preset fallback under Capture On is unchanged: children keep one chain and segment.
- With AGENTS.md project context, an ordinary captured switch from llama.cpp to groq is refused with `unsupported_surface_change`; without it, the switch goes through. Pre-existing; found by probe.
- `test_console_continuation_group_merge.py` (10), `test_console_named_agent_model.py` (2), `test_console_redirect_stream_cut.py` (3) and `test_console_request_provenance_routes.py` (2) are red identically on the base.
- TASK-34100.2.1's fallback half is met by this task.
<!-- SECTION:NOTES:END -->
