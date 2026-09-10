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
- [ ] #1 Tools mode and the Permissions matrix list the built-in server's tools under their own server row with the correct effective state.
- [ ] #2 A Space-cycle on such a row changes the state the next Console approval round resolves.
- [ ] #3 An empty catalog states why it is empty.
- [ ] #4 Tests cover the built-in inventory path in the hub.
<!-- AC:END -->
