---
id: TASK-26042
title: >-
  MCP: wire server-initiated sampling/elicitation to the live chat provider and
  approval surface
status: To Do
assignee: []
created_date: '2026-09-01 23:28'
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
- [ ] #1 A server-initiated sampling request is fulfilled through the live chat provider (chat_api_call) and returned
- [ ] #2 A server-initiated elicitation request is presented through the live approval surface and the response returned
- [ ] #3 Per-server SamplingPolicy (allow + rate + token caps) is sourced from config, default deny
- [ ] #4 client._server_request_dispatcher is set where MCPClient is created, so stdio and (future) remote servers both route through it
<!-- AC:END -->
