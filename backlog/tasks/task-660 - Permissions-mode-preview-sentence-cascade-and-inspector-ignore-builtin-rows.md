---
id: TASK-660
title: Permissions-mode preview sentence, cascade, and inspector ignore agent:builtin rows
status: To Do
assignee: []
created_date: '2026-07-25'
labels: [tools, security, ux, mcp, bug]
dependencies: [TASK-627]
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**Defect**, found during TASK-627 review. TASK-627 appended an `agent:builtin` section to the MCP workbench's Permissions-mode matrix (`UI/MCP_Modules/mcp_workbench.py`), but two adjacent surfaces were never updated to know built-in rows exist, and both now give the user wrong information about the exact state TASK-627 just made settable.

**1. The inspector clears instead of explaining a built-in row.** `MCPWorkbench.on_mcp_permissions_mode_row_selected()` resolves the selected row's tool via `self._tool_for(event.server_key, event.tool_name or "")`, which only ever returns a `HubTool` from the live MCP catalog. A built-in row is never a `HubTool`, so `tool` is always `None` for it, which the handler treats identically to a stale/pinned row: `await inspector.show_tool(None)` — wiping whatever the inspector was showing, with no permission explanation rendered for the row the user just clicked. Every MCP tool row gets `show_permission(tool, effective, cascade=...)`; every built-in row gets nothing. Note the cascade map has the identical gap for the same reason: `_last_cascade` is populated only from the `cascade_map` `_build_permission_rows()` builds while walking the MCP catalog, so even a fixed inspector call has no cascade tuple for a built-in tool to pass — `show_permission(..., cascade=None)` already degrades gracefully to a single origin sentence, so the fix can either extend cascade data to built-ins or accept that fallback deliberately; either is acceptable as long as it is a decision, not an accident.

**2. The preview sentence is computed before built-in rows exist, so it drifts from the table it claims to summarize.** In `_sync_permissions_mode()`:

```python
rows, preview, cascade_map = self._build_permission_rows(
    tools, effective=effective, servers_payload=servers_payload, global_state=global_state,
)
...
rows = rows + self._builtin_permission_matrix_rows(payload, servers_payload)
await self.query_one(MCPPermissionsMode).update_matrix(
    rows, kill_switch=kill_switch, preview=preview, echo=echo
)
```

`preview` is derived by `_build_permission_preview()` **inside** `_build_permission_rows()`, whose `" · N overrides across M servers"` suffix is computed by scanning only the `rows` list that method itself built — i.e. MCP rows only. The built-in section is appended to `rows` *afterward* and is invisible to that scan. `MCPPermissionsMode.update_matrix()`'s own docstring states the preview it renders "ALWAYS summarizes the full, UNFILTERED matrix (`rows`/`echo` as given)" — that invariant is now false for the override count: `rows` (the parameter `update_matrix` actually receives) includes built-in rows, but `preview` (computed earlier, on the pre-merge list) does not.

Net user-visible effect: set a persistent override on a built-in tool (the exact action TASK-627 just enabled) — the table cell correctly shows e.g. `Off •`, but the preview sentence's override count does not change to reflect it, and clicking that row to inspect it clears the inspector instead of explaining the override. The one feature TASK-627 shipped (a settable, inspectable built-in permission) is visible in the table but not corroborated by either adjacent surface.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] Selecting a built-in (`agent:builtin`) row in Permissions mode shows a permission explanation for that row (mirroring `show_permission()` for an MCP tool row), not a cleared inspector
- [ ] A test drives selecting a built-in tool row and asserts the inspector shows that tool's state/origin rather than calling `show_tool(None)`
- [ ] The preview sentence's override count and "across M servers" phrasing include a persistent built-in override when one is set, and a test pins this (set a built-in override, assert the preview sentence changes)
- [ ] `update_matrix()`'s docstring claim that the preview "ALWAYS summarizes the full, UNFILTERED matrix" is either made true again or corrected to state the built-in-section exception explicitly
- [ ] MCP-only behavior is unchanged: an MCP row's inspector selection and the preview sentence for MCP-only overrides render exactly as before this fix
<!-- AC:END -->
