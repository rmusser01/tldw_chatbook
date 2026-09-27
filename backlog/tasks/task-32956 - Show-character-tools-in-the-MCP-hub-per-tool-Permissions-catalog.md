---
id: TASK-32956
title: Show character_* tools in the MCP hub per-tool Permissions catalog
status: Done
assignee:
  - '@claude'
created_date: '2026-09-25 23:14'
updated_date: '2026-09-27 19:19'
labels:
  - mcp
  - characters
dependencies:
  - TASK-32954
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-32954 shipped character_search, character_get and character_save as Console-only local tools. They never appear in the MCP hub's per-tool Tools/Permissions catalog, because the catalog snapshot builds its local tool provider without a character service (the same gap ask_user has). So a user cannot set the two read tools to Allow: with the default Ask, every character search and read raises an approval card, and the Character Creator skill has had to stop telling users they can relax it. The per-tool permission rows should exist so users can choose Allow for the reads while the save keeps asking.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 character_search, character_get and character_save each appear as a row in the MCP hub per-tool Permissions catalog when the character tools gate is on, and none appear when it is off
- [x] #2 Setting character_search or character_get to Allow in the hub lets that tool run in the Console without an approval card
- [x] #3 character_save still raises an approval card on every call even when set to Allow (mutates floor unchanged)
- [x] #4 The built-in Character Creator skill tells the user once that the read tools can be set to Allow in the MCP hub
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace: MCP hub local rows come from UnifiedMCPControlPlaneService.local_hub_tools() -> local_server_tools.build_hub_local_inspection_provider() -> _default_specs() with no character_service, so the character specs never register. Console gate: _compose_local_provider resolve_state -> service.gate_tool_test -> permission_store.resolve_effective_state, keyed (local:__local__, tool name) + definition_hash.
2. Hub rows (AC1): pass an inert CharacterToolService (loader raises 'unavailable') to _default_specs in the hub provider builder; the existing [tools] character_tools_enabled gate keeps them out when off; CONSOLE_ONLY keeps them off the executable/external projection.
3. Save floor (AC3): resolve_effective_state only floors INHERITED allow; an explicit Allow on character_save would run unasked. Add a code-owned always-ask set for (local:__local__, character_save) in permission_store so both the hub row (shows Ask with the floor marker) and the Console gate agree.
4. Skill (AC4): one Permissions line in SKILL.md, re-pin the digest, flip the no-hub-advice pin test.
5. TDD: failing tests first for each AC (real MCPPermissionStore + real service gate + real LocalToolProvider), then docs (User_Guide/mcp.md), ratchets, census, ruff, preflight.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Hub rows: local_server_tools._build_hub_local_provider_handle passes a never-callable CharacterToolService (_hub_character_service; loader raises 'unavailable') to _default_specs, so character_search/get/save register in the Hub inspection catalog whenever [tools] character_tools_enabled is on. They stay CONSOLE_ONLY, so the executable/shared projection drops them and Tools mode lists them as non-runnable. Rows use the same spec code as the Console, so the definition_hash a hub Allow stores matches the Console's live tool.

Save floor (deviation from 'mutates floor unchanged'): resolve_effective_state floors only an INHERITED allow. An explicit tool-level Allow on character_save resolved to allow, and the provider ran the save with no card (confirmed by the new test before the fix). Added permission_store.ALWAYS_ASK_TOOLS = {(local:__local__, character_save)}. It floors every allow to ask with risk_floored=True, so the hub row shows Ask with the floor marker and the Console gate raises the card (reason risk_floored, 'changes local data' copy). The shared floor for other tools is unchanged. The inspector notice is now origin-aware (_risk_floored_notice): a floored explicit Allow reads 'Asks on every call, even when set to Allow.'

Skill: SKILL.md gains a one-paragraph Permissions section (tell the user once). Digest re-pinned. The old no-hub-advice pin test is flipped.

ask_user needs no row: it is gate_exempt and never reaches the permission layer.

Tests: Tests/MCP/test_character_tool_permission_rows.py (real service local_hub_tools, real MCPPermissionStore, hub set_tool_state, real ConsoleChatController._compose_local_provider over an in-memory CharactersRAGDB); Tests/Skills/test_builtin_skills.py; the Console-only pin in Tests/UI/test_mcp_workbench.py.
Files: MCP/local_server_tools.py, MCP/permission_store.py, UI/MCP_Modules/mcp_inspector.py, assets/skills/character-creator/SKILL.md, Skills_Interop/builtin_skills.py, Docs/User_Guide/mcp.md, Docs/User_Guide/console/agent-runs-and-tools.md, backlog/docs/lessons-testing-evidence.md.

Follow-ups (not done): the Console card's 'Always allow' option on character_save persists an Allow that is then floored, so the button does nothing; approve_session still skips later save cards in that session; the Tool Packs permission inventory (Tool_Packs/catalog_snapshot.py) does not list the character tools, so an export reports their stored rules as omitted.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
