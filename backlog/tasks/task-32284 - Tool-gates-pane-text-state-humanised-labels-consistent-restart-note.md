---
id: TASK-32284
title: 'Tool gates pane: text state, humanised labels, consistent restart note'
status: Done
assignee: []
created_date: '2026-09-10 19:14'
updated_date: '2026-09-11 03:51'
labels:
  - mcp
  - tool-gates
  - accessibility
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Servers, Tool gates checkboxes carry state only by colour (Textual's ToggleButton always draws X; toggling read_file flipped the config while the glyph did not change). Labels are raw ids such as read_file while the first-run wizard shows 'Read file' with a description; the pane says changes apply on the next app restart while Tools mode says the next Console agent run. Source: live approval-card / MCP-permissions UX review 2026-09-10 against dev 3315241674 (snapshot .impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each gate row states on or off in text, following the kill-switch label pattern.
- [x] #2 Gate rows use the wizard's humanised names and descriptions from one shared table.
- [x] #3 The restart or next-run note is accurate per gate.
- [x] #4 The Permissions legend's gate-off count links to this pane.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace the gate table, the enumerator, the pane builder and the wizard consumer; measure (not assume) when a gate change actually takes effect.
2. Write failing tests for: copy on every `_GATEABLE_BUILTINS` row, the wizard rendering it, the pane's on/off text labels and click round trip, the per-gate apply note, and the breadcrumb naming the pane.
3. Move the wizard's `_TOOL_COPY` onto `GateableTool` (title/blurb, required); carry them through `ToolGate`.
4. Replace the pane's compact Checkboxes with kill-switch-style toggle Buttons; route presses through `on_button_pressed`.
5. Correct the notes and the breadcrumb; update `Docs/User_Guide/mcp.md` and CLAUDE.md's New Tool checklist.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The Tool gates rows in MCP > Servers > built-in row were compact Checkboxes, whose glyph is X in BOTH states (Textual's ToggleButton draws the same _button either way; only the colour differs), labelled with the raw tool id, under a blanket 'Applies on next app restart' note. All three are fixed from one place. GateableTool now carries required title and blurb fields holding the copy that used to live in the first-run wizard's private _TOOL_COPY dict (moved verbatim, wizard table deleted, ToolsStep reads the row); ToolGate carries that title plus a restart_required flag, and a builtin row's description is now the blurb rather than the LLM-facing description read off a constructed Tool - which also removes the tool construction and its 'Unavailable on this system.' degrade path from all_tool_gates(). The pane renders each gate as a toggle Button labelled '{title}: on/off >' (the _kill_switch_label() idiom, tooltip = blurb), routed through on_button_pressed against self._tool_gates_by_id, so state is carried in text and the label follows the value that was actually saved. The restart note was simply wrong: BuiltinToolProvider/LocalToolProvider are built once per Console agent run and read these keys fresh (verified in-process by flipping read_file_enabled and web_deep_search_enabled and rebuilding), so the note now reads 'Applies to the next Console agent run', with a second note naming the one genuine exception - web_deep_search is also published to external MCP clients by build_server_local_provider(), which runs when the built-in server starts, so that half follows the next client launch. The Permissions legend's breadcrumb now ends with the shared TOOL_GATES_PANE_PATH ('MCP > Servers > built-in row > Tool gates') instead of 'the built-in server detail'. Files: Agents/tool_catalog.py, Agents/builtin_tool_gate.py, UI/MCP_Modules/mcp_servers_mode.py, UI/Wizards/FirstRunSetupWizard.py, Docs/User_Guide/mcp.md, CLAUDE.md, Docs/security/production-diagnostic-inventory.json (one removed warning), and tests in Tests/Agents/test_gateable_builtin_tools.py, Tests/Agents/test_builtin_tool_gate.py, Tests/UI/test_mcp_servers_mode.py, Tests/UI/test_mcp_workbench.py, Tests/Wizards/test_first_run_setup_wizard.py.

Fix round (this task, commit 3a59f4e1e0 on fix/approval-wave-c-hub, on top of f6e2007b4a): the shipped diff's own restart-note claim and citations were themselves imprecise. Fixed: (1) config.py's two web_deep_search_enabled comments still said 'requires app restart' -- reworded to say the Console picks the change up on its next agent run, with external MCP clients seeing it on their next client launch only when [mcp] expose_local_tools is on. (2) The pane's 'Also published to external MCP clients' note rendered unconditionally whenever a gate was flagged restart_required, implying external publication regardless of [mcp] expose_local_tools (default off) -- named that precondition in the sentence instead of adding a second config read plus render-state tests; Docs/User_Guide/mcp.md's caveat paragraph mirrors the wording. (3) Three docstring/comment citations of a nonexistent build_console_tool_registry() (mcp_servers_mode.py, builtin_tool_gate.py, test_builtin_tool_gate.py) now correctly name Chat/console_agent_bridge.py's _compose_run_registry_and_allowed(), the real per-run composer. Minors taken in the same round: Docs/User_Guide/mcp.md's 'ask_user (the only gate here that is on by default)' was wrong -- the local master switch is also on by default; a one-line comment now explains why test_toggling_tool_gate_button_posts_tool_gate_changed_with_section_and_key drives .press() instead of pilot.click (hit-testing coverage lives in test_mcp_workbench.py's round-trip test); _rebuild_tool_gate_checkboxes() was renamed to _rebuild_tool_gate_buttons() (it rebuilds Buttons, not Checkboxes) along with the three test names that still said checkbox for these rows; and _gate_button() now escapes gate.title with rich.markup.escape before building the label, matching every other title-derived label in this file. Verified: Tests/UI/test_mcp_servers_mode.py + Tests/Agents/test_builtin_tool_gate.py + Tests/Agents/test_gateable_builtin_tools.py (107 passed), Tests/UI/test_mcp_workbench.py (342 passed), Tests/Wizards/test_first_run_setup_wizard.py (390 passed, the same 2 pre-existing Summary-step-route failures plus one flaky retry test confirmed to pass 3/3 in isolation -- neither touched by this diff). ruff error counts unchanged on every touched file. ./scripts/preflight.sh: all derived-artifact checks passed.
<!-- SECTION:NOTES:END -->
