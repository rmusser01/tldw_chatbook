---
id: TASK-2067
title: 'MCP: Escape cancels inline forms and focus moves into forms on open (F-056)'
status: Done
assignee: []
created_date: '2026-08-03 17:24'
updated_date: '2026-08-03 20:46'
labels:
  - ux-review
  - mcp
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
No inline form (profile, import, mutations, delete-confirm, test-tool) binds Escape; focus is never moved into a form on open, so keyboard users must Tab to reach inputs. Evidence: mcp_profile_form.py:126-127 et al. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Escape closes/cancels each inline MCP form
- [x] #2 Opening a form moves focus to its first input
- [x] #3 Tests cover escape and initial focus
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (UI keyboard-interaction fix). Steps: 1. Survey each inline form's cancel/close path (profile form Cancel, import panel close, mutations panel close, delete-confirm disarm, test-tool panel Close) and its first input. 2. RED tests per form: Escape triggers the same path as its Cancel/Close control; opening the form focuses its first input. 3. Implement minimal BINDINGS (escape) on each form widget + focus() on open (on_mount of the form or the host's show path). 4. Run MCP form/workbench/inspector tests + ruff.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Approach: per-form show=False escape BINDINGS wired to the SAME message/path each form's own Cancel/Close control uses (no parallel cancel logic): MCPProfileForm/MCPImportPanel/MCPServerMutationsPanel post their existing Cancelled messages; MCPServersMode.action_disarm_delete reuses disarm_delete() (the 'Keep' path); MCPInspector.action_close_test_panel reuses _close_test_tool_panel() and no-ops when no panel is open. Focus-on-open: each form focuses its first reachable input via call_after_refresh (profile form #mcp-form-id or #mcp-form-command in edit since id is disabled; import TextArea; mutations #mcp-srv-id or #mcp-srv-name; delete arm pair focuses 'Keep' so Enter/Escape both back out safely; test panel focuses the schema form's first Input/Select/TextArea with Close as fallback). Files: tldw_chatbook/UI/MCP_Modules/{mcp_profile_form,mcp_server_mutations,mcp_servers_mode,mcp_inspector}.py; tests: 7 new (escape + initial focus per surface, incl. edit-mode variants) in Tests/UI/test_mcp_profile_form.py, test_mcp_server_mutations.py, test_mcp_servers_mode.py, test_mcp_inspector.py. TDD: 5 tests RED before implementation (2 edit-mode focus tests already passed via Textual auto-focus). Verification: 217 passed (profile_form + server_mutations + servers_mode + inspector + schema_form); 198 passed test_mcp_workbench.py; ruff clean. ADR: not required (UI keyboard interaction). Not done: no Escape handling added for the overview/detail panes themselves (no cancel semantic exists there); delete-confirm focuses 'Keep' rather than 'Confirm delete' deliberately (safe default); commit 95021f25a.
<!-- SECTION:NOTES:END -->
