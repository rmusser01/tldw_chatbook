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
The bug as filed does not exist: the hub reads the same seam the Console provider does (local_service.get_inventory() -> builtin_tools_from_inventory() off the one UnifiedMCPControlPlaneService), verified by execution and live in the app -- Tools mode renders the tldw_chatbook group, Permissions has its own 'Server default - tldw_chatbook' group, and a Space-cycle writes builtin:tldw_chatbook/list_characters to the on-disk store, which effective_tool_states() then resolves to allow. The reported symptom is discoverability: the flat catalog sorts by (server_label, name), so ~40 'Local workspace...'/'Virtual CLI' rows precede 'tldw_chatbook' in a ~25-row viewport, and the rail's server selection scopes the Permissions footer but neither canvas -- left for a rider since it changes behaviour for every server. No production code changed; the one real gap (AC#4) is closed by two new tests in Tests/UI/test_mcp_workbench.py, with RED demonstrated by mutation.

Modified files: `Tests/UI/test_mcp_workbench.py` (two new tests plus an
optional `inventory_names` parameter on the existing built-in-inventory
fixture trio, so no third clone was needed). Full trace, live captures and
mutation evidence:
`.superpowers/sdd/2026-09-10-approval-card-fix-wave/task-13-report.md`.
<!-- SECTION:NOTES:END -->
