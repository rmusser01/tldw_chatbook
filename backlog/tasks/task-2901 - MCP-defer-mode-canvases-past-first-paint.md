---
id: TASK-2901
title: MCP — defer the Audit and Tools mode canvases past first paint
status: Done
assignee: []
created_date: '2026-08-07 02:00'
labels:
  - mcp
  - performance
  - defer-past-first-paint
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Screen survey (task-2725 follow-up): MCP mounts 135 widgets; `MCPAuditMode#mcp-mode-canvas-audit` (28) and `MCPToolsMode#mcp-mode-canvas-tools` (14) arrive hidden — ~31% of the screen deferrable. Both are existing widget classes, so the 2725/2900 pattern applies without extraction. Modest but cheap win; audit the mode-switch path's tolerance for absent canvases first (same discipline as 2725).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- SECTION:ACCEPTANCE_CRITERIA:BEGIN -->
- [x] MCP first paint excludes the hidden mode canvases; every mode is reachable after load.
- [x] Mode switching, permissions, and audit flows keep their existing tests green.
- [x] The compose→load window is covered by verified tolerance (no unguarded queries of the deferred canvases).
<!-- SECTION:ACCEPTANCE_CRITERIA:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fourth defer-past-first-paint application (2725/2900 pattern). Compose keeps only the initial Servers canvas inside the ContentSwitcher; Tools/Permissions/Audit mount as `_reload_guarded`'s FIRST step — before `reload()` pushes data through the pipeline's unguarded `query_one(MCP*Mode)` sites, so their validity holds by ordering. Two screen-specific traps, both caught by tests: (1) **ContentSwitcher hides children from its `current` WATCHER** — late-mounted children arrive visible; all three canvases briefly stacked and pushed the current one's content off-screen (19 mounted-flow tests red with OutOfBounds) — deferred canvases are hidden explicitly at mount; (2) `ContentSwitcher.current` raises for an unmounted id, so `set_mode` stashes a request landing in the pre-mount window (`_pending_deferred_mode`) and `_mount_deferred_canvases` replays it through the normal path (ModeChanged included) — pinned by test. Results: MCP tab switch 1.11s → **0.25–0.26s** live; all four modes cycled live with real content, zero errors; **MCP 9-file surface: 692 passed, 0 failed**. Files: tldw_chatbook/UI/MCP_Modules/mcp_workbench.py, Tests/UI/test_mcp_deferred_canvases.py, Tests/UI/test_mcp_workbench.py (fake gains the new probe).
<!-- SECTION:NOTES:END -->
