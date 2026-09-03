---
id: TASK-26030
title: MCP resources and prompts as agent-callable tools
status: To Do
assignee: []
created_date: '2026-08-31 15:46'
updated_date: '2026-09-01 23:50'
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
DEFERRED (2026-09-01): LOW priority; understood but multi-layer. Seam analysis for a future session:
1. Synthesize up to 4 HubTools per server (list_resources/read_resource/list_prompts/get_prompt) in MCP/hub_tool_catalog.py alongside local_tools_from_record, ONLY when discovery_snapshot carries resources/prompts (the snapshot already stores inventory['resources'/'prompts'] via local_control_service.py:206-207) -> AC#5. Namespace via server_key prefix; T1 dedupe_names in compose_catalog handles AC#6.
2. Mark synthetic tools (a reserved name convention avoids changing the frozen HubTool dataclass across many sites) so MCP/unified_control_plane_service.py execute_hub_tool (:2387 local: branch) routes them to the client's list_resources/read_resource/list_prompts/get_prompt (MCPClient has read_resource(server_id,uri):1106 and get_prompt:1138) instead of execute_external_tool. Gate + execution-log + result spill (TASK-25904) then flow through the existing path -> AC#3/#4.
3. compose_catalog (Agents/mcp_tool_provider.py) already runs local_tools_from_record per record; add the synthetic tools there.
Headless verification is limited (real read_resource needs a live MCP server); test the synthesis (AC#5/#6) + routing dispatch with fakes, like TASK-26029 did.
<!-- SECTION:PLAN:END -->
