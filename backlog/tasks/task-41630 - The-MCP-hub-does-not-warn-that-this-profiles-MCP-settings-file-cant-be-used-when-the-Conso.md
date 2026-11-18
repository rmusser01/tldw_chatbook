---
id: TASK-41630
title: "The MCP hub does not warn that this profile's MCP settings file can't be used when the Console MCP store is unbound"
status: To Do
assignee: []
created_date: 2026-11-17 19:05
labels:
- follow-up
- mcp
- ux
priority: low
references:
- tldw_chatbook/UI/MCP_Modules/mcp_rail.py
- tldw_chatbook/UI/MCP_Modules/mcp_inspector.py
dependencies:
- TASK-41308
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Checked on dev 97f8c58c0c; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** TASK-41308 made the Console Inspector say `MCP: off · this profile's MCP settings file can't be used here` whenever the MCP store is unbound, which is the same predicate that drops MCP from the Send. The MCP hub, where the user configures servers, still shows them with no warning. Someone who adds a server there has no way to learn it will never reach the Console.

**Where.** The hub's rail and inspector (`UI/MCP_Modules/`) read the server list but not the store binding.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The MCP hub shows a warning with the same reason as the Console Inspector row whenever the MCP store is unbound
- [ ] #2 No warning is shown when the store is bound
- [ ] #3 A test mounts the hub with an unbound and a bound store and asserts the warning; it fails before the fix
<!-- AC:END -->
