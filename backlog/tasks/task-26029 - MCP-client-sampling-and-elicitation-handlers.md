---
id: TASK-26029
title: 'MCP client: sampling and elicitation handlers'
status: In Progress
assignee: []
created_date: '2026-08-31 15:46'
updated_date: '2026-09-01 23:29'
labels:
  - mcp
  - interop
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Server-initiated MCP requests are all refused. Verified on origin/dev: MCP/client.py:744,754 answers ping and returns -32601 method-not-found for everything else, so a server that asks the client to run a completion (sampling/createMessage) or to ask the user a question (elicitation/create) simply cannot work. This silently narrows which MCP servers are usable. Chatbook has both halves already: a chat provider for sampling and an approval-card surface for elicitation - the gap is the two handlers wiring them to the protocol.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A server-initiated sampling request is fulfilled through the existing chat provider and its result returned to the server
- [x] #2 Sampling requests are gated: the user controls whether a given server may request completions, and the default is not silent consent
- [x] #3 Sampling requests are bounded (rate and token budget) so a server cannot drain the user's account
- [ ] #4 A server-initiated elicitation request is presented to the user through the existing approval-style surface and the response returned
- [x] #5 Elicitation requests that ask for credentials or free-form secrets are refused rather than presented
- [x] #6 A declined or timed-out request returns a well-formed protocol error, never a hang
- [x] #7 Servers that never use these methods are unaffected
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Pure sampling policy (gate + rate + token budget) and elicitation secret-screen\n2. ServerRequestDispatcher with injected complete_fn/elicit_fn; well-formed errors for refuse/decline/timeout\n3. Wire dispatcher into _StdioJSONRPCConnection._handle_server_request (default None -> -32601)\n4. TDD handler + connection dispatch\n5. App-level injection of real chat provider + approval surface = follow-up (app-context)
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Protocol-correct, fail-closed handler + connection dispatch shipped and tested. Production injection of the LIVE chat provider + approval surface is split to TASK-27019 (app-context, not headless-verifiable), so this stays In Progress with ACs #1/#4 pending that wiring.

Shipped (MCP/server_request_handlers.py, new):
- evaluate_sampling_request: pure gate (deny unless explicitly allowed - AC#2) + rate cap + token budget (AC#3).
- screen_elicitation_for_secrets: refuses an elicitation whose prompt or requestedSchema field name/format/description is credential/secret-shaped (AC#5).
- ServerRequestDispatcher: routes sampling/createMessage through an injected complete_fn and elicitation/create through an injected elicit_fn; returns a well-formed JsonRpcError (never raises across the wire) for refuse/decline/provider-failure so a server never hangs (AC#6); unknown methods -> -32601.
- Wired into MCP/client.py _StdioJSONRPCConnection._handle_server_request via an injected dispatcher; MCPClient holds an optional _server_request_dispatcher passed to each connection. Default None keeps the prior method-not-found behavior, so servers not using these methods (or a client without wiring) are unaffected (AC#7).

Tested (Tests/MCP/test_server_request_handlers.py, 16): gate/rate/budget, secret refusal (prompt + schema), dispatcher fulfill/refuse/decline/timeout, and connection-level routing (wired -> reply, unwired -> -32601). Existing MCP client suite kept green after adding the new optional kwarg to 9 test-double Session constructors.

Remaining (TASK-27019): construct the dispatcher with chat_api_call (complete_fn) and the approval-store round-trip (elicit_fn), source SamplingPolicy from config, and set client._server_request_dispatcher at MCP/local_control_service.py:778.

Files: tldw_chatbook/MCP/server_request_handlers.py (new), tldw_chatbook/MCP/client.py, Tests/MCP/test_server_request_handlers.py, Tests/MCP/test_client_catalog_pagination.py (test-double signature).
<!-- SECTION:NOTES:END -->
