---
id: TASK-2901
title: MCP — defer the Audit and Tools mode canvases past first paint
status: To Do
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
- [ ] MCP first paint excludes the hidden mode canvases; every mode is reachable after load.
- [ ] Mode switching, permissions, and audit flows keep their existing tests green.
- [ ] The compose→load window is covered by verified tolerance (no unguarded queries of the deferred canvases).
<!-- SECTION:ACCEPTANCE_CRITERIA:END -->
