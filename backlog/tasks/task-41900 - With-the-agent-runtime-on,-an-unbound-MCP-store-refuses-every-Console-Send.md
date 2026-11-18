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
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the round-16 Console stream (TASK-41507/41511). With the agent runtime on (the Console default), a Send whose MCP store is not bound refuses with `mcp_source_not_installed`, raised in `_AgentSources.__init__` (`Agents/native_source_storage.py`). TASK-40714's guarantee that an ordinary Send reaches the provider holds only because its tests turn the agent runtime off: turning it on fails `test_ordinary_send_reaches…[no-pinned-io|non-default-data-dir]`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An ordinary Send with the agent runtime on and no MCP store bound reaches the provider (or shows a visible, correct refusal that names the missing MCP setup)
- [ ] #2 TASK-40714's ordinary-send tests run with the agent runtime on, red before the fix
<!-- AC:END -->
