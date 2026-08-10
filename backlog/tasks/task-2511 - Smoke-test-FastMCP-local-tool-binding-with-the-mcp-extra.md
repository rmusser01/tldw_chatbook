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
- [ ] #1 Server starts with expose_local_tools=true and lists the local tools,fs_read callable after grant, fs_write refused with EXTERNAL_NO_CALLBACK_REFUSAL,Findings recorded (including any FastMCP 2.x API drift)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Superseded as obsolete by TASK-2512. The standalone server no longer binds
through FastMCP or the official MCP SDK, so this task's proposed smoke cannot
exercise the shipped boundary. TASK-2512 replaces it with independent wheel
and sdist installs of `tldw_chatbook[mcp]`; each artifact launches
`python -m tldw_chatbook.MCP` from site-packages only and verifies the real
`mcp-unified` stdio catalog, local-tool refusal, resources, prompts, isolated
state, metadata, and clean shutdown.
<!-- SECTION:NOTES:END -->
