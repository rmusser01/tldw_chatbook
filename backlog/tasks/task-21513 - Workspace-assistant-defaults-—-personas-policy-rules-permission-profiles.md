---
id: TASK-21513
title: 'Workspace assistant defaults — personas, policy rules, permission profiles'
status: Done
assignee: []
created_date: '2026-08-29 22:59'
updated_date: '2026-08-29 23:06'
labels:
  - console
  - personas
  - workspaces
dependencies: []
priority: high
---

## Description
<!-- SECTION:DESCRIPTION:BEGIN -->
Per-workspace default agent persona with narrowing-only tool policy rules and named permission profiles, unified with tldw_server's Workspace Assistant Defaults contract.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Workspace assistant_defaults stored/read with server-shaped validation and effective resolution with reason codes
<!-- AC:END -->

## Implementation Notes

**Approach.** Implemented the tldw_server `assistant_defaults` contract (server-shaped validation, stored-vs-effective split with reason codes) plus narrowing-only persona policy rules and named permission profiles wired through the existing MCP/Console seams. All prior gates and floors apply first; persona policy only narrows.

**Key decisions.**
- Profile-major inheritance: permission profiles inherit unset keys from the `default` profile.
- Allowlist posture for the default workspace assistant with an editor warning surface.
- Convenience auto-create of missing profiles/persona bindings is non-fatal (degrades silently, logged).
- MCP tool invoke threads the resolved persona policy (`RunToolPolicy` at the registry choke point) rather than pre-filtering at advertisement only.

**Main new/modified modules.**
- New: `tldw_chatbook/Workspaces/assistant_defaults.py`, `tldw_chatbook/Workspaces/agent_provisioning.py`, `tldw_chatbook/Agents/persona_policy.py`, `tldw_chatbook/Agents/run_tool_policy.py`, `tldw_chatbook/Widgets/Persona_Widgets/personas_policy_rules_editor.py`, plus the `workspaces_v6_to_v7_assistant_defaults.sql` migration.
- Touched seams: `DB/Workspace_DB.py`, `Workspaces/{models,registry_service}.py`, `Agents/{tool_catalog,mcp_tool_provider,builtin_tool_gate}.py`, `MCP/{permission_store,unified_control_plane_service,local_server_tools}.py`, `Chat/{console_turn_context,console_chat_controller,console_agent_bridge,console_session_settings}.py`, `UI/Console_Modules/session.py`, `UI/Screens/{settings_screen,personas_screen}.py`, `Widgets/Console/console_workspace_switcher_modal.py`, `Widgets/Persona_Widgets/{persona_profile_editor_widget,personas_inspector_pane,personas_persona_visual_pack_widget}.py`, `Persona_Visual/importer.py`, `Character_Chat/local_character_persona_service.py`, `tldw_api/character_persona_schemas.py`, `app.py`.

**Governance.** ADR `backlog/decisions/079-workspace-assistant-defaults.md`; spec `Docs/superpowers/specs/2026-08-29-workspace-assistant-defaults-design.md` (Status: Implemented). Verified: targeted sweep 603 passed / 0 failed; `py_compile` clean on all touched production Python files.
