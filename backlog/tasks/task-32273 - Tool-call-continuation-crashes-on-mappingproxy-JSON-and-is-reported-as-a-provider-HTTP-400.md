---
id: TASK-32273
title: >-
  Tool-call continuation crashes on mappingproxy JSON and is reported as a
  provider HTTP 400
status: Done
assignee: []
created_date: '2026-09-10 19:09'
updated_date: '2026-09-10 20:24'
labels:
  - agents
  - llm-calls
  - bug
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After a native tool call, the next provider request carries a MappingProxyType (ToolCall.arguments became immutable in 4ed4757d53, 2026-09-04) and the requests-based OpenAI-compatible handler cannot serialise it. The failure is shown to the user as 'provider returned HTTP 400 ... The provider rejected this request. Confirm the model is still available', blaming the provider for a client-side bug. Reproduced live with the custom provider; other handlers unverified. Source: live approval-card / MCP-permissions UX review 2026-09-10 against dev 3315241674 (snapshot .impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A multi-step tool-call turn (find_tools, load_tools, tool) completes through the custom/OpenAI-compatible handler without a serialisation error.
- [x] #2 Every provider handler that re-sends tool-call history serialises immutable argument mappings; a regression test covers the serialisation seam.
- [x] #3 A client-side serialisation failure is reported to the user as an app error, never as a provider rejection or HTTP status.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause was not the tool-call builder: every builder already emits plain dicts with a string `function.arguments` (verified live with probes at console_agent_bridge and agent_service). The frozen rows are injected at the trace seam -- ConsoleTraceService issues `freeze_json`-frozen `messages_payload` and `ConsoleProviderGateway._trace_surface_kwargs` puts those exact objects on the adapter kwargs, because the surface verifier requires identity between what was recorded and what is dispatched. A row of scalars survived `json.dumps`; the first native tool-call continuation row (nested `tool_calls` mappings) did not, so `requests` raised TypeError in request preparation. Fix: new `adapter_wire_kwargs()` in console_provider_gateway thaws `messages_payload` back to mutable JSON containers AFTER verification and immediately before `_enter_provider_adapter`, so identity verification still passes and every provider handler gets serialisable rows from the one seam. Second half: `_chat_with_openai_compatible_local_server`'s `(ValueError, KeyError, TypeError)` branch raised `ChatBadRequestError(400)` -- nothing there came back from the provider -- so it now raises `ChatConfigurationError(status_code=None)` (that class gained an optional status_code, defaulting to 500 as before); a status-less configuration error makes `_provider_error_copy_with_model_recovery` skip the 'The provider rejected this request' copy and `safe_provider_error_copy` omit 'Status:', and `describe_stream_failure` now summarises it as 'the app could not build the request'. Real provider 4xx still arrive as requests' HTTPError and keep their true status. Live-verified against the repo fake LLM server: find_tools -> load_tools -> mcp__tldw_chatbook__list_characters and the approval card renders. Files: Chat/console_provider_gateway.py, Chat/provider_failures.py, Chat/Chat_Deps.py, LLM_Calls/LLM_API_Calls_Local.py, Tests/Chat/test_console_provider_gateway.py, Tests/LLM_Calls/test_custom_openai_credential_resolution.py.
<!-- SECTION:NOTES:END -->
