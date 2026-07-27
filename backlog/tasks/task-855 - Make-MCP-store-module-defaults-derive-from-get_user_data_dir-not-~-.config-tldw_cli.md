---
id: TASK-855
title: >-
  Make MCP store module defaults derive from get_user_data_dir(), not
  ~/.config/tldw_cli
status: To Do
assignee: []
created_date: '2026-07-27 04:34'
labels:
  - security
  - mcp
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
MCP/local_store.py:13 (DEFAULT_LOCAL_MCP_STORE_PATH), MCP/unified_context_store.py:14-16, and MCP/server_target_store.py:13 all default to DEFAULT_CONFIG_PATH.parent / <name> -- i.e. ~/.config/tldw_cli/. Every real construction site instead passes get_user_data_dir() / <name> explicitly (app.py:5241, :5248, :4028), so today nothing actually uses the module defaults. This is rated latent rather than live because of what those defaults would produce if ever hit: MCP/unified_control_plane_service.py:2430 derives the permission-store path as Path(store.path).with_name("mcp_permissions.json"), and :2073 derives the execution-log path the same way from store.path. A LocalMCPStore() built with no explicit argument anywhere in the codebase, in a test, or in a future call site would silently place the permission store and execution log in ~/.config/tldw_cli/ -- a location neither _sensitive_single_file_paths() (which joins to get_user_data_dir()) nor the parent == user_data_dir rule in Utils/sensitive_paths.py covers. A reproduction confirmed is_sensitive_path() returns False for both ~/.config/tldw_cli/mcp_permissions.json and ~/.config/tldw_cli/local_mcp_store.json.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The three MCP store modules' default paths derive from get_user_data_dir(), or the classes require an explicit path argument with no config-relative fallback
- [ ] #2 A test constructs each store with no explicit path and asserts the resulting path matches the get_user_data_dir()-derived location (or that construction without an explicit path is disallowed), not a hardcoded ~/.config/tldw_cli literal
- [ ] #3 Every current explicit-path construction site (app.py:5241, :5248, :4028) continues to resolve to the same paths as before
<!-- AC:END -->
