---
id: TASK-41905
title: "An MCP deep link can fail at teardown on the workbench's canvas-delete disarm"
status: To Do
assignee: []
created_date: 2026-11-18 20:31
updated_date: 2026-11-18 20:31
labels:
- follow-up
- ui
- mcp
priority: medium
references:
- tldw_chatbook/UI/MCP_Modules/mcp_workbench.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41506 (round 16). `MCPWorkbench._disarm_canvas_delete` queries `MCPServersMode` after awaiting `_sync_children_lock`, with the default `exit_on_error`, and raised `WorkerFailed NoMatches` at teardown in 1 of 44 runs of `test_mcp_deep_link_with_explicit_context_lands_on_tools_hub`. That node and `test_tools_settings_deep_link_with_profile_payload_lands` stay off the UI lane until it is fixed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The disarm queries nothing once its screen is detached or pruning, and raises nothing
- [ ] #2 Both deep-link nodes run on the UI lane and pass 10 of 10 at `-n 4`
<!-- AC:END -->
