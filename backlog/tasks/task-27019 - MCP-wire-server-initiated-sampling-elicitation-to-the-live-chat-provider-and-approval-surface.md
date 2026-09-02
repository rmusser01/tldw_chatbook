---
id: TASK-27019
title: >-
  MCP: wire server-initiated sampling/elicitation to the live chat provider and
  approval surface
status: In Progress
assignee: []
created_date: '2026-09-01 23:28'
updated_date: '2026-09-02 13:45'
labels:
  - mcp
  - interop
dependencies:
  - TASK-26029
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-26029 shipped the protocol-correct handler + connection dispatch for server-initiated sampling/elicitation, fail-closed and tested with injected callables. Production enablement requires the app to construct the dispatcher with the REAL chat provider (chat_api_call) as complete_fn and the REAL approval surface (async create-request then await resolution) as elicit_fn, source the per-server SamplingPolicy from config, and set client._server_request_dispatcher at the MCPClient creation site (MCP/local_control_service.py:778). This half is app-context and not headless-verifiable, so it was split out.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A server-initiated sampling request is fulfilled through the live chat provider (chat_api_call) and returned
- [ ] #2 A server-initiated elicitation request is presented through the live approval surface and the response returned
- [x] #3 Per-server SamplingPolicy (allow + rate + token caps) is sourced from config, default deny
- [x] #4 client._server_request_dispatcher is set where MCPClient is created, so stdio and (future) remote servers both route through it
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Live wiring shipped (MCP/live_server_request_wiring.py + client factory + creation-site hookup); one honest residue keeps this In Progress.

- AC#1 DONE: sampling runs through the real chat provider — chat_api_call off-loop, non-streaming, bounded max_tokens; provider/model from flat [mcp] sampling_provider/sampling_model falling back to [chat_defaults] provider/model; MCP message shapes converted; provider-shape-tolerant text extraction (eval-runner pattern).
- AC#3 DONE: per-server SamplingPolicy from config, DEFAULT DENY — a server samples only if listed in [mcp] sampling_allowed_servers; caps from sampling_max_requests_per_minute (6) / sampling_max_total_tokens (50k).
- AC#4 DONE: MCPClient gains _server_request_dispatcher_factory, called with the server_id at connect time (per-server policy; sampling budgets persist per server across reconnects); set at the single creation site (LocalMCPControlService._get_client). Factory failure degrades to method-not-found, never breaks connect.
- AC#2 PARTIAL: elicitation is live at the SERVICE layer as a confirmation through the hub approval-request store (pending request w/ fingerprint; approved -> accept, denied -> None, timeout -> expired + TimeoutError so an abandoned request can never be approved into a void; complex field-value schemas refused up front — never show a prompt whose answer we cannot represent). The control plane exposes approval_requests.list/approve/deny, but NO MCP-screen view drives them yet — the user-visible list is the residue (small UI addition + live verify).

Tests: Tests/MCP/test_live_server_request_wiring.py (10: default-deny + allowlist/caps, chat_api_call conversion/extraction + model-hint precedence, approve/deny/timeout against a real store file, complex-schema refusal, per-server factory budgets, creation-site wiring). MCP suite: 1089 pass, only the 41 dev-inherited baseline failures.

Files: tldw_chatbook/MCP/live_server_request_wiring.py (new), tldw_chatbook/MCP/client.py, tldw_chatbook/MCP/local_control_service.py, Tests/MCP/test_live_server_request_wiring.py.
<!-- SECTION:NOTES:END -->
