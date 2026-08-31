---
id: TASK-26030
title: MCP resources and prompts as agent-callable tools
status: To Do
assignee: []
created_date: '2026-08-31 15:46'
labels:
  - mcp
  - agents
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The agent cannot reach MCP resources or prompts. Verified on origin/dev: the client implements them - MCP/client.py:1044 read_resource and :1076 get_prompt - and the control plane surfaces them (MCP/unified_control_plane_service.py:1177,1183), but Agents/mcp_tool_provider.py:307,653 composes a tools-only catalog, so a server exposing its useful content as resources is invisible to the model. Hermes synthesizes list_resources, read_resource, list_prompts and get_prompt as callable tools per server.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The agent can list and read resources from a connected MCP server
- [ ] #2 The agent can list and fetch prompts from a connected MCP server
- [ ] #3 These synthetic tools pass through the same permission gate and execution log as the server's real tools
- [ ] #4 Resource content is subject to the same size ceiling and spill behavior as any other tool result
- [ ] #5 Servers exposing no resources or prompts do not gain empty tools that clutter the catalog
- [ ] #6 The synthetic tools are namespaced per server so two servers cannot collide
<!-- AC:END -->
