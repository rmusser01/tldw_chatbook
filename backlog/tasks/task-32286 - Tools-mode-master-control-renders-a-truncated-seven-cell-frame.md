---
id: TASK-32286
title: Tools-mode master control renders a truncated seven-cell frame
status: Done
assignee:
  - '@claude'
created_date: '2026-09-10 19:15'
updated_date: '2026-09-11 04:57'
labels:
  - mcp
  - tools-mode
  - ui
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The 'Local workspace, web, and Watchlists tools' control at the top of Tools mode renders as a tiny bordered frame showing a truncated checkbox beside an 'Enabled' label at 250 columns. Source: live approval-card / MCP-permissions UX review 2026-09-10 against dev 3315241674 (snapshot .impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The master control renders its checkbox and label fully at 80 to 250 columns.
- [x] #2 A real-bundle CSS harness test pins the control's width.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Measure the control's region and find the culprit CSS rule in a real-bundle harness at 80/120/250 columns.
2. Write failing tests asserting the full label renders untruncated within the control's region at those widths (TDD RED).
3. Convert the Checkbox + separate Enabled/Disabled Static pair to one toggle Button carrying the state in its label text, matching mcp_servers_mode._gate_button()'s task-32284 kill-switch idiom -- retires the bundle's width:8 escape hatch instead of widening it.
4. Rebuild the CSS bundle; update the existing Checkbox-typed tests in test_mcp_tools_mode.py and test_mcp_workbench.py to the new Button contract, including the round-trip save/read-back test.
5. Run the affected test files, preflight, close the task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Measured the control at 80/120/250 columns before touching code: the Checkbox + separate Enabled/Disabled Static pair was clamped by the bundle's MCPToolsMode #mcp-tools-local-enabled { width: 8 } escape hatch (mcp_tools_mode.py BUNDLED_CSS, plus a load-bearing hand-authored copy in css/components/_agentic_terminal.tcss that CSS_PATH rules need to outrank widget DEFAULT_CSS) to a bordered 7-cell frame, truncating the Checkbox's own On label to a lone ellipsis; tracing the history showed the app-wide unscoped Checkbox { width: 100%; height: 2 } rule that escape hatch was fighting was itself retired in TASK-18960, leaving the width:8 override fighting nothing. Wrote failing real-bundle tests first (RED, verified via a scoped git stash of only the source+CSS files) asserting the full label renders untruncated within the control's own region at 80/120/250 columns, then replaced the Checkbox+title+state trio with one toggle Button whose label states its own on/off in text, reusing mcp_servers_mode._gate_button()'s task-32284 kill-switch idiom for the exact same [console] local_tools_enabled gate (already duplicated there as the Servers-mode Tool gates master switch) -- a Button sizes to its own content so the escape hatch is retired outright rather than widened. Updated the five pre-existing tests across test_mcp_tools_mode.py and test_mcp_workbench.py that queried the old Checkbox/title/state ids, including the master-switch save/read-back round-trip contract (toggle.press() now drives the save, and update_local_config() re-renders the Button label from persisted truth). Rebuilt the CSS bundle (build_css.py) and confirmed both derived-file diffs only remove the retired selectors. Full test_mcp_tools_mode.py (33 passed) and test_mcp_workbench.py (343 passed, matching the pre-change baseline exactly) are green; the unrelated 39-failure baseline in Tests/MCP/test_mcp_documentation_contract.py was confirmed pre-existing via the same stash technique. preflight.sh passes. Updated Docs/User_Guide/mcp.md with one sentence on the new toggle-button form plus a docs-pass stamp line. Files changed: tldw_chatbook/UI/MCP_Modules/mcp_tools_mode.py, tldw_chatbook/css/components/_agentic_terminal.tcss, tldw_chatbook/css/tldw_cli_modular.tcss, tldw_chatbook/css/widget_defaults_scoped.tcss, tldw_chatbook/css/widget_defaults_self.tcss, Tests/UI/test_mcp_tools_mode.py, Tests/UI/test_mcp_workbench.py, Docs/User_Guide/mcp.md.
<!-- SECTION:NOTES:END -->
