---
id: TASK-42525
title: "A Console agent's provider fallback is refused at dispatch when Capture is On"
status: In Progress
assignee:
- '@claude'
created_date: 2026-11-23 23:00
updated_date: 2026-11-25 03:03
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
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With Capture On, an agent run whose primary provider fails switches to a ready configured fallback provider and completes, as it does with Capture Off
- [x] #2 The captured trace shows the failed calls and the fallback call, and which provider each went to
- [x] #3 A retry after a failed call that changes the provider, model or endpoint without being a fallback is still refused; the existing `changed_*` cases are still refused
- [x] #4 No call is left in `reserved` after a refused or completed fallback
- [x] #5 A test drives a Capture-On agent run through a fallback; it is red before the change and green after
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
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Status: AC#1 is met only in part; it needs an owner (ADR) choice (see "AC#1 and its limits" below).** AC#2-#5 are met.

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

**AC#1 and its limits.** AC#1 holds when the fallback's request continues the run's trace surface unchanged. That covers the round-24 probe's deepseek -> groq shape, credit and exhausted, before and after tool calls, and a native -> text-tool provider before any tool call. Two cases still stop on "Trace capture could not start" before the fallback provider is asked, with nothing reserved; Capture Off completes them:
- **Re-encoded tool history.** The model wrote its tool calls as text and the fallback provider takes them natively, or the reverse. ADR-110's projection rewrites rows already on the surface (`test_a_fallback_that_must_rewrite_the_runs_tool_history_is_refused`).
- **A different system-prompt placement.** This is TASK-42472's path. llama.cpp is sent its system prompt as the first message and a hosted provider is not, so a llama.cpp run's fallback to groq is refused even before any tool call (`test_a_captured_llamacpp_run_cannot_yet_fall_back_to_a_hosted_provider`, with and without project context).

Both need the new chain's first call to replace part of the surface. The amendment grants no replacement, and the 2026-09-08 per-call rendered-system amendment says it adds no replacement permission, so this is an owner (ADR) choice. Recommendation: let a fallback chain's first call carry one bounded replacement. Its removed nodes would be the surface's leading rendered-system slot and the run's earned tool rows, each passing the completed-turn witness's run-lineage checks, within the 256-node bound.

**Other limits and finds (proposed follow-ups).**
- A spawned child's preset fallback under Capture On is unchanged: children keep one chain and segment.
- Found while probing (pre-existing, not this change): with AGENTS.md project context, an ordinary next send that switches a captured chat from llama.cpp to groq is refused with `unsupported_surface_change`; without project context it is sent. The probe was deleted.
- TASK-34100.2.1's fallback half (its AC#2 and the fallback part of AC#4) is met by this task.
<!-- SECTION:NOTES:END -->
