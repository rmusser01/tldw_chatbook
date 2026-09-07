---
id: TASK-2066
title: >-
  MCP: footer shortcut bar stops advertising keys that do not work in context
  (F-055)
status: Done
assignee: []
created_date: '2026-08-03 17:24'
updated_date: '2026-08-03 18:19'
labels:
  - ux-review
  - mcp
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Footer shows 'space cycle permission' in all four modes but the key only works in Permissions with the matrix focused; pressing t in Servers mode force-switches to Tools and notifies 'Select a tool first.' Evidence: mcp_screen.py:30-41. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Footer shortcut context is per-mode and only shows working keys
- [x] #2 t with no tool selected is a no-op with a hint (no mode hijack)
- [x] #3 Tests updated
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Approach: (1) mcp_screen.py: _COMMON_SHORTCUTS (1-4/a/r, the screen-level bindings that work in every mode) + MCP_MODE_SHORTCUTS per-mode map (t only in Tools, space only in Permissions); _register_footer_shortcuts(mode) resolves the active mode and re-registers on mount, resume, and every MCPWorkbench.ModeChanged. (2) mcp_workbench.py open_test_for_selected_tool(): opens the panel first via inspector.open_test_panel() and set_mode('tools') only on 'opened'; 'no_tool' now notifies 'Select a tool in Tools mode first.' (warning) with NO mode switch -- the hijack is gone. Files: tldw_chatbook/UI/Screens/mcp_screen.py, tldw_chatbook/UI/MCP_Modules/mcp_workbench.py; tests: test_destination_shells.py (2 footer tests updated to per-mode expectations + new test_mcp_destination_footer_shortcuts_follow_mode cycling all four modes; stale MCP_SHORTCUTS docstring reference updated), test_mcp_workbench.py (no-selection test now asserts mode stays 'servers' + hint copy). TDD: 4 tests RED before implementation. Verification: 302 passed + 1 skip (test_destination_shells.py + test_mcp_workbench.py); 10 passed (phase6 recovery, 2 MCP geometry parity tests, footer context suite); ruff clean. ADR: not required (UI shortcut-hint + keybinding behavior fix, no contract change). Not done: 'a'/'r' remain advertised in all four modes (they genuinely work everywhere); the space binding stays display-only on MCPPermissionsMode (its matrix must own focus) -- only its advertising is now mode-correct; commit ec6ac4d32.
<!-- SECTION:NOTES:END -->
