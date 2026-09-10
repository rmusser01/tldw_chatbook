---
id: TASK-32283
title: Built-in server tools are missing from Tools mode and the Permissions matrix
status: To Do
assignee: []
created_date: '2026-09-10 19:14'
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

1. Trace the hub's built-in inventory read against the Console provider's read of the same seam, with executable evidence.
2. Reproduce live before changing anything.
3. Add the regression coverage the hub half of the seam lacks.
4. Verify: named test files + a live pass over Tools and Permissions.

## Implementation Notes

**The bug as filed does not exist; no production code changed.** The hub already
reads the same seam the Console provider reads: both do
`getattr(service, "local_service", None).get_inventory()` ->
`builtin_tools_from_inventory()` off the one
`UnifiedMCPControlPlaneService` the app builds (`app.py:10997`). All three
hypotheses in the brief were falsified by direct execution against the worktree
package: `get_inventory()` returns 30 tools from a static manifest (no session
needed), `_collect_hub_tools()` driven off a real service returns 30
`builtin:tldw_chatbook` HubTools, and the `mcp.inventory.list.local` gate
evaluates to allowed (hence the absent warning -- nothing raised).

What the reviewer saw is a discoverability failure, reproduced live:
`MCPToolsMode._apply_filter()` sorts the flat cross-server catalog by
`(server_label, name)`, so "Local workspace, web, and Watchlists" and
"Virtual CLI (read-only)" put ~40 rows ahead of the "tldw_chatbook" group in a
~25-row viewport, and the rail's server selection does not scope either canvas.
Filtering or scrolling reveals the rows immediately. Verified live that the
Permissions matrix has its own `Server default - tldw_chatbook` group (AC#1),
that one Space on `list_characters` writes
`servers["builtin:tldw_chatbook"]["tools"]["list_characters"]["state"]="allow"`
to the on-disk store and re-renders both canvases as `Allow` (AC#2), and that
`_empty_tools_diagnosis()` already explains an empty catalog (AC#3).

AC#4 was the one real gap: nothing pinned the hub half of the seam at row level.
Added two tests to `Tests/UI/test_mcp_workbench.py` -- built-in inventory ->
`#mcp-tools-table` rows keyed `builtin:tldw_chatbook::<name>`, and a
Space-cycle round trip through the real store plus a real
`effective_tool_states()` resolve. Since the behaviour is already correct, RED
was demonstrated by mutation: disabling the inventory branch fails both tests
with `KeyError/ValueError: 'builtin:tldw_chatbook::list_characters'`.
The existing built-in-inventory fixture trio gained an optional
`inventory_names` parameter rather than a third near-identical clone.

The brief's conditional fix (Refresh-tools-triggers-discovery plus the
"Built-in tools appear after the first Console tool call or Refresh tools."
empty state) was deliberately NOT built: its precondition is false, so it would
be unreachable code.

Follow-up recommended (not in scope here, changes behaviour for every server):
the rail's selected server scopes the Permissions footer summary but not the
matrix or the Tools table above it, which is the contradiction that produced
this misfiled bug.

Modified files: `Tests/UI/test_mcp_workbench.py`.
Full trace, live captures and mutation evidence:
`.superpowers/sdd/2026-09-10-approval-card-fix-wave/task-13-report.md`.
