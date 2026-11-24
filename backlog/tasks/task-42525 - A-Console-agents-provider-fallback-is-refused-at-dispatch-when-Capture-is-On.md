---
id: TASK-42525
title: "A Console agent's provider fallback is refused at dispatch when Capture is On"
status: To Do
assignee: []
created_date: 2026-11-23 23:00
updated_date: 2026-11-23 23:00
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
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the round-24 review of TASK-42472 (2026-11-23) and reproduced on fu/r24-diag (dev b31e3e9241 plus round 24). Paths are under `tldw_chatbook/` unless they start with `Tests/`.

**Why.** Capture is on by default for saved chats, and with Capture On a user's configured provider fallback never runs. When an agent's model call fails, `Agents/agent_runtime.py::_recover_model_failure` first retries in place. After the retries are spent (a transient error), or at once (a 402/403 credit-terminal error, or `ChatModelUnavailableError` for a preset target), it switches to the next ready fallback provider. Users set that chain in `[console] agent_fallback_providers` (`Chat/console_agent_bridge.py::console_fallback_providers`, wired into `AgentConfig.fallback_providers`) or with preset fallback targets. The switched call is the run's next call, so it goes out as a TOOL_LOOP call with a different provider, model and endpoint.

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
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With Capture On, an agent run whose primary provider fails switches to a ready configured fallback provider and completes, as it does with Capture Off
- [ ] #2 The captured trace shows the failed calls and the fallback call, and which provider each went to
- [ ] #3 A retry after a failed call that changes the provider, model or endpoint without being a fallback is still refused (the existing `changed_*` cases stay red-to-refusal)
- [ ] #4 No call is left in `reserved` after a refused or completed fallback
- [ ] #5 A test drives a Capture-On agent run through a fallback; it is red before the change and green after
<!-- AC:END -->
