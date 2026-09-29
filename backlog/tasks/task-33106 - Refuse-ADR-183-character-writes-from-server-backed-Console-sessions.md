---
id: TASK-33106
title: Refuse ADR-183 character writes from server-backed Console sessions
status: Done
assignee:
  - '@claude'
created_date: '2026-09-27 17:30'
updated_date: '2026-09-29 16:44'
labels:
  - mcp
  - characters
dependencies:
  - TASK-32955
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-32955 (PR #2863) added a server-mode refusal to ADR-183's `create_character` and `update_character`, but only for the standalone MCP server. That server has no Console session, so it reads the default profile's runtime source. In-process calls, where a Console agent reaches ADR-183's tools through the MCP bridge, deliberately keep dev's behaviour. The default-profile check could have refused a local session or let a server-backed session write.

So a Console agent in a server-backed session can still write LOCAL character cards through the bridge, while the Console's own `character_save` refuses in that case. The in-process path should refuse using the calling session's own runtime source, not the default profile's.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A Console agent in a server-backed session that calls `create_character` or `update_character` through the in-process MCP bridge is refused with the same message the Console character tools use.
- [x] #2 A Console agent in a local session keeps working as today.
- [x] #3 The standalone MCP server's behaviour (TASK-32955) is unchanged.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. MCPToolProvider (per-run, Console-composed) gains an optional runtime_source_provider. _execute refuses ADR-183's built-in character writes (BUILTIN_MCP_SERVER_KEY + CHARACTER_WRITE_TOOLS) with the Console's SERVER_REFUSAL when the run's session is server-backed. The audit row keeps its authorizing decision with the refusal as error. A raising lookup fails closed.
2. ConsoleChatController._compose_mcp_provider passes a session-bound _session_runtime_source, the same method character_save's wiring now uses. No session: no check.
3. Standalone server (MCP/server.py, MCPTools) unchanged.
4. Tests: provider unit tests (server refused / local + unbound run / raising fails closed); Console composition test (server session refused end to end, local reports local, no session no check). Failure-name comparison vs dev for touched files.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The Console composes a per-run MCPToolProvider, so that provider now carries the calling session's runtime source. A new optional runtime_source_provider is read at execution time. _execute refuses ADR-183's built-in character writes (BUILTIN_MCP_SERVER_KEY + CHARACTER_WRITE_TOOLS) with the Console character tools' own SERVER_REFUSAL when the session is server-backed, before anything reaches the in-process runtime. The audit row keeps the decision the call was authorized under, with the refusal as its error. A raising session lookup fails closed. _execute is the single point every allowed path passes (stamp, allow state, session approval, arg rule, fresh card), so no approval route bypasses the check.

ConsoleChatController gains _session_runtime_source(session_id), passed to MCPToolProvider by _compose_mcp_provider (no session: no check, today's behaviour). character_save's wiring now uses the same method instead of its private copy. SERVER_REFUSAL is imported lazily because Tools.character_tool_service is not in the UI-ready census; MCP.builtin_tool_policy already is.

The standalone MCP server (MCP/server.py, MCPTools) is unchanged (AC#3); MCP/tools.py only gains a comment pointing at the new in-process check.

Tests: test_mcp_tool_provider.py (server refused with the audit row and reads unaffected; local and unbound runs write; a raising lookup fails closed), and test_console_character_wiring.py (the composed provider reads the calling session's backend; a server session's create_character is refused end to end; no session applies no check).

Verification: failure names across the 72 test files that touch MCPToolProvider, the Console MCP composition or the character tools match dev exactly (1,589 each; +7 passes). Preflight, the census and ruff counts are unchanged.
<!-- SECTION:NOTES:END -->
