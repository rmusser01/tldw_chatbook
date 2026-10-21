---
id: TASK-35152
title: 'Palette: drop duplicate Switch-to-Settings entry from POPULAR_TABS'
status: Done
created_date: 2026-10-16 20:14
updated_date: 2026-10-21 00:42
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
POPULAR_TABS in tldw_chatbook/app_command_providers.py still lists TAB_MCP; after the MCP destination retirement (task-35150 / ADR-240) both TAB_MCP (now an alias) and TAB_SETTINGS render a Switch-to-Settings command in the palette's popular row. Remove the duplicate or relabel it 'Settings — Tools & MCP'.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Exactly one Settings entry in the popular palette row,Provider test updated and green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: update palette provider/shell-route tests to expect one 'Switch to Settings' + one 'Switch to Tools & MCP' in the popular row, the folded-route deeplink table, truthful 'tools'/'mcp' alias searches, and refreshed MCP help copy. 2. Generalize LIBRARY_SUBROUTE_COMMANDS to FOLDED_ROUTE_COMMANDS (the artifacts label mechanism) and add the 'mcp' entry 'Tab Navigation: Switch to Tools & MCP' (routing unchanged: mcp -> Settings {category: tools}); refresh the two stale TAB_HELP_TEXT strings. 3. GREEN on Tests/UI/test_command_palette_providers.py + test_command_palette_shell_routes.py, then lint.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Approach: kept the popular-row slot for the retired MCP route (per ADR-240's Tools & MCP fast-path intent) but gave it a distinct label via the same mechanism Artifacts uses — an explicit deep-link command table entry. Generalized `TabNavigationProvider.LIBRARY_SUBROUTE_COMMANDS` to `FOLDED_ROUTE_COMMANDS` (it now holds a non-Library entry) and added `("mcp", "Tab Navigation: Switch to Tools & MCP", "Open Settings → Tools & MCP for servers, tools, permissions, and audit")`. `_tab_command`'s existing table lookup makes `discover()` render the distinct label; `search()`'s existing table loop makes "tools"/"mcp" queries surface the deep-link that actually lands on Settings' Tools category. Routing untouched: `route_for_tab(TAB_MCP) == "mcp"` still rides `LEGACY_ROUTE_NAV_CONTEXTS["mcp"] = {"category": "tools"}`.
- Refreshed the two stale help strings in `TAB_HELP_TEXT`: TAB_MCP → "Open Settings → Tools & MCP for servers, tools, permissions, and audit"; TAB_TOOLS_SETTINGS → "Open Settings → Tools & MCP (legacy tools-settings route)".
- Tests (TDD, red→green): `Tests/UI/test_command_palette_providers.py` popular row now asserts exactly one "Switch to Settings" + one "Switch to Tools & MCP" among 6 hits; the dedup test counts the new command exactly once; the legacy-alias search table truthfully expects "Switch to Tools & MCP" for "tools"/"mcp". `Tests/UI/test_command_palette_shell_routes.py` pins the new help copy and adds `test_palette_offers_tools_and_mcp_deeplink_command` (label + unchanged routing); renamed the two `LIBRARY_SUBROUTE_COMMANDS` references.
- Modified files: `tldw_chatbook/app_command_providers.py`, `Tests/UI/test_command_palette_providers.py`, `Tests/UI/test_command_palette_shell_routes.py`.
- ADR required: no — direct implementation of existing ADR-240 label policy (ADR-240 already governs the MCP retirement; this only fixes the palette label/copy).
- Verification: `pytest Tests/UI/test_command_palette_providers.py Tests/UI/test_command_palette_shell_routes.py -q` → 69 passed; the 4 remaining failures in those files (SettingsProvider/LibraryIngestProvider count drift, skills starter-route) fail identically at HEAD (verified via `git stash`), so they are pre-existing and out of scope. No linter is installed in the worktree venv (no ruff/flake8/black); `py_compile` passes on all changed files.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
