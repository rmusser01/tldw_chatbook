---
id: TASK-855
title: >-
  Make MCP store module defaults derive from get_user_data_dir(), not
  ~/.config/tldw_cli
status: Done
assignee:
  - '@claude'
created_date: '2026-07-27 04:34'
updated_date: '2026-07-27 16:29'
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
- [x] #1 The three MCP store modules' default paths derive from get_user_data_dir(), or the classes require an explicit path argument with no config-relative fallback
- [x] #2 A test constructs each store with no explicit path and asserts the resulting path matches the get_user_data_dir()-derived location (or that construction without an explicit path is disallowed), not a hardcoded ~/.config/tldw_cli literal
- [x] #3 Every current explicit-path construction site (app.py:5241, :5248, :4028) continues to resolve to the same paths as before
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm no real construction site anywhere in the codebase relies on the module-level default (all pass an explicit path).
2. Decide: derive the default from get_user_data_dir() lazily, vs. require an explicit path. Pick lazy derivation (avoids baking a stale profile into a module-level constant at import time, matching the existing lazy pattern in Utils/sensitive_paths.py).
3. Replace the three eager DEFAULT_*_PATH module constants with lazy _default_*_path() helper functions, called from each class's __init__ only when no path is given.
4. Add tests asserting default-construction resolves to get_user_data_dir()-joined paths, that explicit-path sites are unaffected, and that the default is covered by is_sensitive_path().
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Decision: derive lazily rather than require an explicit path. Nothing in the codebase (app.py, or any of the ~90 test call sites across Tests/MCP/ and Tests/RuntimePolicy/) ever constructs these three classes with no argument, so 'require explicit path' would have been equally safe -- but it would make every one of those call sites' intent (an explicit path) implicit and give up a legitimate escape hatch for a future caller that genuinely wants the default. Made the default LAZY (a private _default_*_path() function called from __init__ only when path is falsy) rather than an eager module-level constant computed at get_user_data_dir()'s import time, for two reasons: (1) get_user_data_dir() has side effects (reads live config, creates the directory) that an import-time module constant would trigger merely by importing the module; (2) an eager constant would bake in whichever profile/HOME was active the FIRST time the module was imported in a process (e.g. a test session), silently going stale for every later profile switch -- exactly the staleness class Utils/sensitive_paths.py's own lazy _sensitive_db_paths()/_sensitive_single_file_paths() helpers were written to avoid. Removed the now-unused DEFAULT_LOCAL_MCP_STORE_PATH/DEFAULT_UNIFIED_MCP_CONTEXT_PATH/DEFAULT_SERVER_TARGETS_PATH constants entirely (grepped first: nothing outside their own module referenced them). New tests (Tests/MCP/test_store_default_paths.py) assert: each store's no-argument default equals get_user_data_dir()/<filename>; the default re-resolves correctly after TLDW_CONFIG_PATH is retargeted (proving lazy, not cached); the default (and its permission-store/execution-log derivatives) is covered by is_sensitive_path(); and explicit-path construction is unaffected. Files: tldw_chatbook/MCP/local_store.py, unified_context_store.py, server_target_store.py; Tests/MCP/test_store_default_paths.py (new).
<!-- SECTION:NOTES:END -->
