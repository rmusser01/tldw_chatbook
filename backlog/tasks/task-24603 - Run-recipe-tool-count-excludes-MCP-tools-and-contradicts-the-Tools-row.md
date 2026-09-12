---
id: TASK-24603
title: Run recipe tool count excludes MCP tools and contradicts the Tools row
status: Done
assignee:
  - '@claude'
created_date: '2026-08-30 00:53'
updated_date: '2026-08-30 01:21'
labels:
  - console
  - ux
  - inspector
  - critique-2026-08-29
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
build_console_inspector_state interpolates normalized_tool_count into the Run recipe line while the Tools row uses effective_tool_count = normalized_tool_count + mcp_tool_count. Live, the rail rendered 'Run recipe: ... / tools 0' eight rows above 'Tools: 4 ready' and 'MCP: 4 tools ready'. The TASK-1843 comment directly above documents this same divergence being fixed once on the status chip and once on the Tools row; the recipe line is the third instance.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Run recipe tool count and the Tools row report the same number for the same state
- [x] #2 A test pins the two against one shared derivation so a future divergence fails
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a parametrised RED test asserting the Run recipe tool segment equals the Tools row for five (built-in, mcp) combinations.
2. Point the recipe line at effective_tool_count, the same derivation the Tools row already uses.
3. Re-run the run-inspector and right-rail suites for regressions.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: ConsoleInspectorState.from_values computed both normalized_tool_count (built-ins only) and effective_tool_count (built-ins + MCP catalog), then interpolated the former into the Run recipe line while the Tools row used the latter. Live capture showed 'Run recipe: ... / tools 0' eight rows above 'Tools: 4 ready' / 'MCP: 4 tools ready'.

This is the third instance of the same divergence: the TASK-1843 comment directly above the Tools row records it being fixed once on the status chip and once on the row; the recipe line was missed both times.

Fix is one identifier. The guard is the test: five (tool_count, mcp_tool_count) combinations assert the recipe's tools segment and the Tools row report the same number, so a future divergence fails rather than being noticed by a user. The two zero-MCP cases pass before the fix and are kept deliberately as negative controls -- they prove the test is sensitive to MCP presence specifically.

Modified: tldw_chatbook/Chat/console_display_state.py, Tests/UI/test_console_run_inspector.py.
<!-- SECTION:NOTES:END -->
