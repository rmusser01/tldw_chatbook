---
id: TASK-41900
title: "With the agent runtime on, an unbound MCP store refuses every Console Send"
status: To Do
assignee: []
created_date: 2026-11-18 20:31
updated_date: 2026-11-18 20:31
labels:
- follow-up
- console
- mcp
priority: high
references:
- tldw_chatbook/Agents/native_source_storage.py
- tldw_chatbook/MCP/console_snapshot.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the round-16 Console stream (TASK-41507/41511). With the agent runtime on (the Console default), a Send whose MCP store is not bound refuses with `mcp_source_not_installed`, raised by `_source_identity` (`MCP/console_snapshot.py` ~137), called from `_AgentSources.__init__` (`Agents/native_source_storage.py` ~107). The refusal is silent (no notice). TASK-40714's guarantee that an ordinary Send reaches the provider holds only because its tests turn the agent runtime off: turning it on fails `test_ordinary_send_reaches…[no-pinned-io|non-default-data-dir]`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The notes name which real configurations leave the MCP store unbound at Send time (TASK-41308 called the non-default data folder case unproven in production)
- [ ] #2 With the agent runtime on and the MCP store unbound, the turn runs without MCP tools instead of being refused, as TASK-40714 AC#2 requires
- [ ] #3 TASK-40714's ordinary-send tests run with the agent runtime on, red before the fix
<!-- AC:END -->
