---
id: TASK-2511
title: Smoke-test FastMCP local-tool binding with the mcp extra
status: Done
assignee: []
created_date: '2026-08-06 07:12'
updated_date: '2026-08-10 08:58'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase-4 follow-up (task-2828 notes): the FastMCP binding path in MCP/server.py::_register_local_agent_tools has never executed in CI because the mcp package isn't in the repo venv (binding tests skip). Do a one-time manual smoke run in a throwaway venv with the [mcp] extra: start the server with [mcp] expose_local_tools=true, list tools, call fs_read (granted) and fs_write (refused). Note: binding tests use the FastMCP-1.x mcp._tool_manager API — FastMCP 2.x made list_tools async; tests may need updating then.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 TASK-2512 independently smoke-tests the installed wheel and sdist with the `mcp` extra
- [x] #2 The original FastMCP smoke was not performed because its runtime boundary is obsolete
<!-- AC:END -->

## Implementation Plan

1. Confirm TASK-2512 proves the replacement `mcp-unified` boundary from
   independently installed wheel and sdist artifacts.
2. Record the original FastMCP smoke as obsolete and close this duplicate
   follow-up without claiming that smoke ran.

ADR required: no
ADR path: N/A
Reason: task supersession records an already-decided runtime boundary; it does
not introduce an architectural decision.

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Original FastMCP acceptance criterion: superseded, not completed. No FastMCP
smoke was performed. TASK-2512 replaces it with GREEN independent wheel and
sdist installs of `tldw_chatbook[mcp]` that exercise the shipped `mcp-unified`
stdio boundary from site-packages only.
<!-- SECTION:NOTES:END -->
