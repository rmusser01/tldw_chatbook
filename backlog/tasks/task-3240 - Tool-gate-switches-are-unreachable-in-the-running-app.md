---
id: TASK-3240
title: Tool gate switches are unreachable in the running app
status: To Do
assignee: []
created_date: '2026-08-07 20:15'
labels:
  - settings
  - ux
  - tools
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**Corrected scope (2026-08-07, owner correction — the original filing overclaimed):** the original description said "no tool gate can be flipped from inside the app at all", ignoring the MCP screen. That was wrong about tool management generally: the MCP screen's Tools mode is a live, nav-reachable hub catalog (MCP servers + builtin inventory + the local agent tool group via _local_agent_hub_tools when [console] local_tools_enabled is on), and its Permissions mode sets Allow/Ask/Off per tool against mcp_permissions.json — including builtin and local tools. That is the per-call permission half of the double opt-in, and it works today.

The real gap is one layer narrower: the **[tools] <name>_enabled registration gates** — the config switches that decide whether a tool exists in a catalog at all (the other half of the double opt-in). Their only switch UIs are UI/Tools_Settings_Window.py (deprecated by TASK-1346; the "tools_settings" route resolves to MCPScreen) and the FirstRunSetupWizard (seen once). And because everything the MCP screen shows is downstream of the gates — builtin_permission_rows enumerates a BuiltinToolProvider, which only registers gate-enabled tools; a gate-off web_deep_search is absent from LocalToolProvider._default_specs and therefore from hub_tools — **a gate-off tool is invisible to the entire live surface**: it can't be discovered or enabled in-app after first run, only by hand-editing config.toml.

Fix-shape options (owner decision): (a) a gate affordance in the MCP hub — arguably the natural home, since it already lists local tools and manages the permission layer; e.g. show gate-off tools greyed with an explicit enable action; (b) a Tools category in the canonical settings_screen (under active critique work on fix/settings-ux-critique-rounds — coordinate); (c) re-route "tools_settings" to a live wrapper of the existing window. Restart-to-apply semantics for construction-time gates (web_deep_search) must stay visible in whatever ships.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A user can discover and flip [tools] registration gates (including gates currently OFF) through the running app's navigation after first run
- [ ] #2 All [tools] gate switches (gateable builtins + web_deep_search) are present there and round-trip to config.toml
- [ ] #3 Gates that need an app restart to apply state that where shown
<!-- AC:END -->
