---
id: TASK-32955
title: Expose the character card tools to external MCP clients
status: Done
assignee:
  - '@claude'
created_date: '2026-09-25 18:23'
updated_date: '2026-09-27 22:22'
labels:
  - characters
  - mcp
dependencies:
  - TASK-32954
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-32954 ships the character_search / character_get / character_save tools as Console-only (exposure CONSOLE_ONLY). External MCP clients connecting to Chatbook's MCP server should also be able to search, read, and (with approval) create or update character cards, the way the Watchlists family offers CONSOLE_AND_EXTERNAL_MCP tools. This needs its own decision on how approvals and the per-session truncation guard work for a client that has no Console session.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 External MCP clients can list and call character_search and character_get
- [x] #2 External character writes go through ADR-183's create_character/update_character only with an explicit operator tool-level grant: 'ask' is refused, server-level and global defaults never auto-allow them, and character_save stays Console-only
- [x] #3 The truncation guard (update_character needs a full character_get read of a long field) and the server-mode refusal behave the same for MCP callers as for the Console
- [x] #4 Exposure can be turned off independently of the Console tools
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Extract the character spec builder from _default_specs so the server can build it without the Console gate; re-mark only character_search/character_get as CONSOLE_AND_EXTERNAL_MCP in build_server_local_provider.
2. New [mcp] expose_character_tools gate (default off, coerced), independent of [tools] character_tools_enabled and of [mcp] expose_local_tools; server publishes the two reads over a real LocalCharacterPersonaService with one per-process CharacterReadGuard.
3. ADR-183 writes: server-mode refusal (SERVER_REFUSAL) in MCPTools for both modes; truncation guard on update_character in the standalone server, sharing the guard with the external character_get.
4. Tests with real SQLite/service/permission store: switch off/on, read-then-update guard, server-mode refusals, ask/server-default/global allow do not unlock update_character, Hub rows unchanged.
5. ADR-183 dated amendment, Docs/User_Guide/mcp.md, verification vs clean dev baseline, preflight.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
External MCP clients can now read character cards, and ADR-183's write tools get the Console's truncation guard and server-mode refusal.

- Exposure (AC#1, AC#4): new `[mcp] expose_character_tools` (default off, coerced in `load_settings` and at the gate) publishes `character_search`/`character_get`. It is independent of `[mcp] expose_local_tools` (decision: enabling character reads must not also expose fs/git/web tools; `web_deep_search`'s nested gate is a sub-feature of that family, which this is not) and of `[tools] character_tools_enabled` (the server builds the specs via the new ungated `_character_specs()` extracted from `_default_specs`). `build_server_local_provider` re-marks only those two specs external with `dataclasses.replace`; the Hub-local composition is untouched, so TASK-32956's rows stay Console-only. The reads run over the real `LocalCharacterPersonaService` (`server_character_service()`), behind the ordinary external gate: explicit Allow runs, ask is refused. `character_save` is never published.
- Writes (AC#2): ADR-183's standing-grant model is unchanged; new tests prove ask, a server-level allow, and a global allow all leave create/update at `permission_required`, and only a tool-level Allow runs them.
- Guard (AC#3): when `expose_character_tools` is on, `TldwMCPServer` owns one `CharacterReadGuard` per process (= per stdio session), assigned to `MCPTools.character_read_guard` only after the reads are published, and shared by the external `character_get` and `update_character`. With the switch off (or a failed publication) no guard applies, so ADR-183's long-field updates are unchanged (coordinator ruling). `update_character` returns `read_full_field_first` (Console message and thresholds, via the new shared `first_unread_long_field`, which `character_save` now uses too). In-process runtime: no guard (no read tool could satisfy it; its Hub/approval-card gates remain) -- documented in the ADR.
- Server mode (AC#3): `MCPTools._write_character` refuses with `unsupported` + `SERVER_REFUSAL` when `load_runtime_source()` (new, normalizes `load_default_runtime_source_state`) says server -- both runtimes. The external reads refuse via `CharacterToolService` as in the Console. A raising loader fails the write closed (storage_error).
- Docs: ADR-183 dated amendment; `Docs/User_Guide/mcp.md` new "Character reads and the long-field guard" section + stamp; `Docs/Design/MCP.md` config keys; config template line.
- Files: Agents/local_tool_provider.py, MCP/local_server_tools.py, MCP/server.py, MCP/tools.py, Tools/character_tool_service.py, config.py; Tests/MCP/test_character_external_mcp.py (new), Tests/test_config_mcp_defaults.py, Tests/MCP/test_mcp_documentation_contract.py (pinned sentence updated), Tests/MCP/test_mcp_unified_stdio.py (its compose harness pins the new gate off like expose_local_tools, and its provider fake accepts the new keyword options). `MCPTools.character_read_guard` is a class-level default so `MCPTools.__new__` harnesses keep working.
- Known, pre-existing and left by design: an in-process `update_character` reached through the Console's MCP bridge with an explicit tool-level Allow has no approval card and no guard (ADR-183's explicit-grant model).
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
