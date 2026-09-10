---
id: TASK-32283
title: Built-in server tools are missing from Tools mode and the Permissions matrix
status: Done
assignee: []
created_date: '2026-09-10 19:14'
updated_date: '2026-09-10 20:09'
labels:
  - mcp
  - permissions
  - bug
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After approving builtin:tldw_chatbook list_characters in Console, MCP Tools and Permissions list only the 'Local workspace, web, and Watchlists' server's 30 tools; Refresh tools and Open tool catalog change nothing. The hub's built-in inventory read yields nothing while the Console provider's read of the same service returns the tools. Users cannot pre-configure the tool that just asked them. Source: live approval-card / MCP-permissions UX review 2026-09-10 against dev 3315241674 (snapshot .impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Tools mode and the Permissions matrix list the built-in server's tools under their own server row with the correct effective state.
- [x] #2 A Space-cycle on such a row changes the state the next Console approval round resolves.
- [x] #3 An empty catalog states why it is empty.
- [x] #4 Tests cover the built-in inventory path in the hub.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace the hub's built-in inventory read against the Console provider's read of the same seam, with executable evidence.
2. Reproduce live before changing anything.
3. Add the regression coverage the hub half of the seam lacks.
4. Verify: named test files + a live pass over Tools and Permissions.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Real cause: ordering, not inventory.** The hub always read the same seam the
Console provider reads (`service.local_service.get_inventory()` ->
`builtin_tools_from_inventory()`, one shared `UnifiedMCPControlPlaneService`),
and the built-in server's 30 tools were always in the catalog with correct
effective state. What hid them was the flat sort: `MCPToolsMode._apply_filter()`
ordered by `(server_label, name)` and `MCPWorkbench._build_permission_rows()` by
`(server_label, key)`, so "Local workspace, web, and Watchlists" and "Virtual CLI
(read-only)" put ~40 rows ahead of "tldw_chatbook" in a ~25-row viewport. The
rail's server selection scoped only the Permissions footer summary, so selecting
the built-in server produced a footer reading "tldw_chatbook: 0 allow · 30 ask"
above rows describing a different server -- which is what got this filed as a
missing-inventory bug. "Open tool catalog" compounded it by switching modes
without carrying the server across.

**Change.** Both sorts gain a leading `key != selected` term, so the
rail-selected server's group comes first and everything else keeps its existing
relative order; with no selection the term is constant and the order is
byte-identical to before. `MCPToolsMode.update_tools()` takes the rail selection
as `selected_server_key`, and a new `MCPToolsMode.focus_server()` (the filter
Select's own mechanism, reused) lets the `OPEN_TOOL_CATALOG` hub action land in
Tools mode already scoped to the server the inspector was showing. No new
scoping concept and no other behaviour change: the filter Select, the footer,
and the rail all keep their current meanings.

**Tests.** Kept the two regression tests from the first round (built-in
inventory -> `#mcp-tools-table` rows; Space-cycle round trip through the real
store plus a real `effective_tool_states()` resolve, RED shown by mutation).
Added three: canvas-level ordering in `Tests/UI/test_mcp_tools_mode.py`, matrix
ordering and the "Open tool catalog" drill in `Tests/UI/test_mcp_workbench.py`
(that is where `_selected_server_key` and the matrix row order actually live --
`MCPPermissionsMode` only renders rows it is handed). All three were red first,
each on its own missing behaviour.

**Live.** Fresh scratch profile: clicking "Open tool catalog" on the built-in
server's inspector lands in Tools mode with the Select reading `tldw_chatbook`
and only that server's rows; Permissions now opens with `Server default —
tldw_chatbook` directly under `Global default`; and with the filter back on "All
servers" the built-in group still leads the table.

Modified files: `tldw_chatbook/UI/MCP_Modules/mcp_tools_mode.py`,
`tldw_chatbook/UI/MCP_Modules/mcp_workbench.py`,
`Tests/UI/test_mcp_tools_mode.py`, `Tests/UI/test_mcp_workbench.py`.
Full trace, live captures and red/green evidence:
`.superpowers/sdd/2026-09-10-approval-card-fix-wave/task-13-report.md`.
<!-- SECTION:NOTES:END -->
