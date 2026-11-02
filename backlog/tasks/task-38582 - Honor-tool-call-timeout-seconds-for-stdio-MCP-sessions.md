---
id: TASK-38582
title: Honor tool_call_timeout_seconds for stdio MCP sessions
status: In Progress
created_date: 2026-11-02 14:26
labels:
- bug
- mcp
priority: high
updated_date: 2026-11-02 14:50
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The stdio transport hardcodes a 10 second wire timeout ignoring the configured knob and kills the server process on any slower tool call, bricking the session and hiding the real cause as transport unavailable. See bug-sweep-2026-11-02-dev.md section 1.3.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Configured timeout reaches the stdio session request deadline
- [x] #2 A tool call slower than 10 seconds completes instead of killing the server
- [x] #3 Teardown happens only on transport uncertainty not plain deadlines
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Approach: `MCPClient.connect_to_server`/`connect_profile` gained an
  optional `request_timeout_seconds` parameter; when omitted it resolves
  ``[mcp] tool_call_timeout_seconds`` via the same settings helper the
  control plane's `_tool_call_timeout` uses (default 60.0). The lazy
  `tldw_chatbook.config` import keeps the boot ratchet intact (ADR-097), and
  an unreadable knob degrades to the default instead of failing the connect
  (availability-only setting; the control plane's own timeout read still
  fails loudly on the execution path). `_REQUEST_TIMEOUT_SECONDS` (10 s)
  stays the class-level default for direct `_StdioJSONRPCConnection`
  constructions, per the batch ambiguity resolution.
- Deadline-without-death: `request()`'s timeout branch now checks
  `_deadline_transport_healthy()` (reader up, no cleanup, child alive,
  stdin writable); a deadline on a provably live transport raises the typed
  `TimeoutError` for THAT request only -- `_settle_failed_request` teardown
  stays reserved for transport uncertainty, and the best-effort
  `notifications/cancelled` is still sent first. `call_tool` maps the wire
  `TimeoutError` to the existing `mcp_tool_timeout` failure kind (the same
  kind the control plane's outer timeout reports), keeping the dispatch
  observation honestly `uncertain` since the server may have run the tool.
- Tests (`Tests/MCP/test_stdio_request_timeout.py` + new fixture
  `Tests/MCP/fixtures/stdio_timeout_server.py`, both real subprocess,
  children reaped in teardown): slow tool past a 0.5 s session deadline ->
  `mcp_tool_timeout`, server still alive (returncode None), a subsequent
  fast call on the same session succeeds; genuine process kill mid-call
  still tears down (`mcp_transport_unavailable`); config plumbing reaches
  the session (42.5 via config, 7.0 explicit override) and the helper
  degrades to 60.0 on unparsable/unreadable settings. Pre-fix run: 3 failed
  (slow-tool returned `mcp_transport_unavailable` with the server killed,
  plumbing and helper tests had no code to exercise); post-fix all 4 pass.
  The 9 fake-session constructors in
  `Tests/MCP/test_client_catalog_pagination.py` gained the new kwarg
  (all 144 still pass).
- Files: `tldw_chatbook/MCP/client.py`,
  `Tests/MCP/test_stdio_request_timeout.py`,
  `Tests/MCP/fixtures/stdio_timeout_server.py`,
  `Tests/MCP/test_client_catalog_pagination.py`.
- ADR required: no -- bug fix inside the documented MCP client contract;
  `Docs/Architecture/mcp-hub.md`'s "Tool execution timeout -> bounded
  refuse" row now actually holds for the stdio wire layer.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
