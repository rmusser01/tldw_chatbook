---
id: TASK-41308
title: "When the MCP store is unbound, a Send runs without MCP tools and neither the user nor the model is told"
status: To Do
created_date: 2026-11-16 19:55
labels:
- follow-up
- console
- mcp
- windows
- ux
priority: medium
references:
- tldw_chatbook/Chat/console_display_state.py
- tldw_chatbook/MCP/console_snapshot.py
- tldw_chatbook/Backup_Recovery/mcp_source_participants.py
- https://github.com/rmusser01/tldw_chatbook/pull/3541
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the 2026-11-13..16 burn-down. TASK-40714 (PR #3541) found it and left it unfixed. Checked on dev 5e3c8b2bd4 (2026-11-16) by reading the code; line numbers are that tree's, and paths are under `tldw_chatbook/`.

**Why.** TASK-40714 made an unbound MCP store count as "no MCP source": the Send goes through with an empty MCP maximum, so the model is offered no MCP tools. The store is unbound on every Windows install and with a non-default data directory. A user with MCP servers configured gets replies that silently ignore those tools, and nothing says why. The model is not told either, so it may claim the tools do not exist.

**Evidence.**
- `Backup_Recovery/mcp_source_participants.py:105-111`: the store's bind returns early without pinned I/O, so a Windows store is never bound.
- `MCP/console_snapshot.py:284` (`unbound_console_sources`) accepts such a store as a source with no MCP tools.
- The inspector's MCP row, `_mcp_inspector_row` (`Chat/console_display_state.py:222-255`, called at `:1992`), shows only "N tools ready" or "M servers enabled, not connected". It returns `None`, so the row disappears, when the count is `None` or both counts are zero. It has no "unavailable, and why" state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 When the MCP store is unbound and MCP servers are configured, the Console shows the user that MCP tools are off for this Send and why (e.g. in the inspector's MCP row), shown by a test that is red on dev
- [ ] #2 The model is told, in the request, that MCP tools are unavailable for this turn, or the task notes record why it should not be
- [ ] #3 With a bound store, the MCP row and the request are unchanged (existing tests still pass)
<!-- AC:END -->
