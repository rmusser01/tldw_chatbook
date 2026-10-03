---
id: TASK-32369
title: anthropic and cohere handlers emit real HTTP 400 errors for local validation failures
status: Done
assignee:
  - '@claude'
created_date: '2026-09-11 01:55'
labels:
  - chat
  - providers
dependencies: []
priority: low
---

## Description

While fixing the misattributed "provider returned HTTP 400" on the custom-openai path (task-32342), lane A found that the anthropic and cohere handlers in `LLM_Calls/LLM_API_Calls.py` (around L1569 and L2604) raise real 400-shaped errors for failures that never left the client (local validation), so Console again blames the provider.

Source: approval-card / MCP-permissions fix wave 2026-09-10/11 (plan `Docs/superpowers/plans/2026-09-10-approval-card-fix-wave.md`, review snapshot `.impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md`); rider recorded in the lane ledger, not fixed in the wave.

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Anthropic ("no valid user messages") and Cohere ("no user/assistant/tool messages") raise ChatConfigurationError(status_code=None) naming the field, instead of ChatBadRequestError(400).
2. ChatConfigurationError carries an optional field name; safe_provider_error_copy says a status-less configuration error was not sent and names that field.
3. The Console stream path stops rewrapping a status-less configuration error as a provider 502 (the non-stream path already re-raises it, task-32342).
4. Tests: one per handler, plus a gateway stream test with the real Anthropic handler checking the Console copy.
<!-- SECTION:PLAN:END -->

## Acceptance Criteria

- [x] Local validation failures in the anthropic and cohere handlers raise `ChatConfigurationError` (status_code None) or an equivalent client-side error, not an HTTP 400
- [x] Console's provider-failure copy for those cases says the request was rejected locally, naming the field
- [x] One test per handler pins the local-failure path

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The two local checks (Anthropic: no user message; Cohere: no user/assistant/tool message besides the system prompt) now raise `ChatConfigurationError(status_code=None, field="messages")` with a constant message, instead of `ChatBadRequestError` (status 400). `ChatConfigurationError` gained an optional `field`.
Console copy: `safe_provider_error_copy` turns a status-less configuration error that names a field into "Request to <provider> not sent: it failed a local check on <field>." It shows only the field name (checked with `str.isidentifier`), never the message, because task-32342's message carries raw detail. A status-less error without a field keeps its old copy: task-32342 also raises it for a reply that could not be read, after the request was sent, so "not sent" would be false there. A configuration error the provider did answer keeps "Provider error from ...".
Streaming gap found while tracing: the stream consumer turned every worker error into `ChatProviderError(..., 502)`, so even task-32342's status-less error read "provider returned HTTP 502" when streamed (only the non-stream path re-raised it). The queue error item now carries `local`, and the consumer re-raises a status-less `ChatConfigurationError`, which `describe_stream_failure` already reports as "the app could not build the request or read the reply".
Tests (`Tests/LLM_Calls/test_local_validation_errors.py`, bootstrap profile because the handlers read settings before the check): one per real handler; the copy; and the real Anthropic handler through `ConsoleProviderGateway.stream_chat`. The handlers' session factory is replaced with one that fails the test, so a regression cannot reach a real API with the test key. Four mutations (Anthropic status back to 400, Cohere field dropped, stream ignoring `local`, copy branch off) each turn a test red. User Guide: a troubleshooting bullet in console/chat-basics.md. Lesson recorded in backlog/docs/lessons-console-wiring.md (two exits).
Files: Chat/Chat_Deps.py, Chat/console_provider_gateway.py, LLM_Calls/LLM_API_Calls.py, Tests/LLM_Calls/test_local_validation_errors.py, Docs/User_Guide/console/chat-basics.md.
<!-- SECTION:NOTES:END -->
