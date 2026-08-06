---
id: TASK-2511
title: Smoke-test FastMCP local-tool binding with the mcp extra
status: To Do
assignee: []
created_date: '2026-08-06 07:12'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase-4 follow-up (task-1345 notes): the FastMCP binding path in MCP/server.py::_register_local_agent_tools has never executed in CI because the mcp package isn't in the repo venv (binding tests skip). Do a one-time manual smoke run in a throwaway venv with the [mcp] extra: start the server with [mcp] expose_local_tools=true, list tools, call fs_read (granted) and fs_write (refused). Note: binding tests use the FastMCP-1.x mcp._tool_manager API — FastMCP 2.x made list_tools async; tests may need updating then.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Server starts with expose_local_tools=true and lists the local tools,fs_read callable after grant, fs_write refused with EXTERNAL_NO_CALLBACK_REFUSAL,Findings recorded (including any FastMCP 2.x API drift)
<!-- AC:END -->
