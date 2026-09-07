---
id: TASK-2539
title: '"Server-source tools are display-only." message is unpinned at its raise site'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-06 09:48'
updated_date: '2026-08-06 18:12'
labels:
  - mcp
  - honesty
  - tests
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR-T3 Task 3's refusal classifier (`mcp_workbench.py`, `_is_permission_refusal()`)
matches a specific `ValueError` by exact string —
`_SERVER_SOURCE_DISPLAY_ONLY_MESSAGE = "Server-source tools are display-only."` — to
render it as `Blocked · not run` instead of `Failed`. The string is raised at
`unified_control_plane_service.py:2235` (inside `execute_hub_tool()`, for a
server-source key).

Nothing pins that literal AT ITS RAISE SITE. The UI-side classifier has a test
asserting its own constant, but no test in the `unified_control_plane_service` /
`execute_hub_tool` suite asserts the exact text the production code actually raises.
A future reword of that message — even a small tidy-up unrelated to this PR — would
silently break the string-match in `_is_permission_refusal()`, and the refusal would
revert to rendering `Failed`. Nothing in the existing suite would catch it: both
sides currently pass independently, and neither test reads the other's string.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A test in the `execute_hub_tool` / `unified_control_plane_service` test suite
      asserts the exact message raised for a server-source tool execution attempt.
- [x] #2 Either that test asserts the same literal `_is_permission_refusal()` matches,
      or a single shared constant is introduced that both the raise site and the UI
      classifier import (preventing future drift structurally, not just by test
      coverage).
- [~] #3 No behavior change to the refusal path itself — this is a coverage/drift-proofing
      fix only. SUPERSEDED: fix-round-B item 3 combined this with task-2537's separate
      defect into one typed-refusal redesign, which intentionally changes matching from
      message-string comparison to exception-type comparison. See Implementation Notes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a dedicated MCPServerSourceDisplayOnlyError(ValueError) exception in unified_control_plane_service.py, raised at execute_hub_tool()'s server-source branch with the byte-identical message.
2. Narrow _is_permission_refusal() to match the typed exception, not the message string.
3. Add a drift-proofing test at the raise site pinning type + exact message.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Went further than either AC #2 option: instead of a shared string constant, execute_hub_tool()'s server-source branch now raises a dedicated MCPServerSourceDisplayOnlyError(ValueError) (unified_control_plane_service.py), and mcp_workbench._is_permission_refusal() matches that TYPE, not the message text at all -- structurally immune to any future reword, stronger than AC #2's own two listed options. Message stays byte-identical to what the bare ValueError carried (the type's own default), so every render-side pin listed in the fix-round brief (Tests/UI/test_mcp_workbench.py:3283/3300, test_mcp_inspector.py:1478) passed unchanged. Drift-proofing test added at the raise site: test_hub_tool_unknown_prefix_raises_the_typed_display_only_error (Tests/MCP/test_control_plane_tool_execute.py), pinning both the type and the exact string. AC #3 ("no behavior change") is marked superseded, not done: this task was folded into fix-round-B item 3's combined design with task-2537, which deliberately changes _is_permission_refusal() from message-based to type-based matching -- a real, intentional behavior change (a bare ValueError with the matching text, from anywhere else, no longer classifies as a refusal; only the typed exception does). Files: tldw_chatbook/MCP/unified_control_plane_service.py, tldw_chatbook/UI/MCP_Modules/mcp_workbench.py, Tests/MCP/test_control_plane_tool_execute.py.
<!-- SECTION:NOTES:END -->
