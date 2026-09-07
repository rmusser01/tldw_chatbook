---
id: TASK-2838
title: Surface the local agent tool catalog in the MCP Hub Tools and Permissions modes
status: Done
assignee:
  - '@kimi'
created_date: '2026-08-06 17:50'
updated_date: '2026-08-30 16:32'
labels:
  - mcp
  - agents
  - hub
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The local agent tool set (fs_*, git_*, web_* — 15 catalog tools, `todo_write` excluded as Console-session-scoped) shipped on dev for the Console runtime and, behind `[mcp] expose_local_tools`, for external stdio clients — but the MCP Hub screen still shows only the 10 legacy built-in tools. The Hub Tools-mode catalog is assembled from three sources (external profiles, built-in AST-derived inventory, remote server payloads) and the local provider was deliberately kept out of the built-in manifest so its AST extraction stayed stable, leaving the Hub with no source for these tools at all. Users therefore cannot see, inspect, or manage the local tool set from the Hub even though permission state for it lives in the same shared `mcp_permissions.json` store the Hub already reads and writes. This task wires the local catalog into the Hub as a proper fourth source so the Tools and Permissions modes reflect reality; Hub-side execution (Test Tool panel) is a deliberate follow-up.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 Hub Tools mode (local source) lists every registered local agent tool under a "Local workspace" group keyed `local:__local__` whenever `[console] local_tools_enabled` is on (the same master opt-in the Console composition applies), with descriptions, input schemas, and risk tags carried from the provider's own HubTool view; `todo_write` remains absent (Console-session-scoped)
- [x] #2 The Tools-mode State column and the Permissions-mode matrix resolve each local tool's On/Off/Ask state from the shared permission store, and a state change written from Permissions mode persists under `local:__local__` so the Console agent runtime honors it on its next run (and vice versa)
- [x] #3 Local tools render as non-executable in the Hub (inspector "not_executable" state, no Test Tool affordance) until Hub-side execution is wired; no raw `tools/call` path to them is opened by this change
- [x] #4 A failure while building the local tool view (provider construction, root resolution) degrades to "no local group" with a warning log and never breaks or empties the existing profile/built-in catalog
- [x] #5 Automated tests cover the new derivation, the catalog assembly inclusion, the fail-soft path, and permission-state round-tripping for a `local:__local__` tool
<!-- AC:END -->

## Implementation Notes

- Approach: the Hub catalog gained its fourth source by reusing the provider's own `HubTool` view rather than inventing a parallel projection. `Agents/local_tool_provider.py` grew a `hub_tools()` method (registration-order list of the existing `hub_tool_for()` views — the exact payload the permission store fingerprints). `UI/MCP_Modules/mcp_workbench.py` `_collect_hub_tools()` now appends `_local_agent_hub_tools()` in its local-source branch: a catalog-view `LocalToolProvider` (workspace root via `resolve_server_workspace_root()`; no `todo_store`, so `todo_write` stays absent; no approval callbacks) mapped to `executable=False` via `dataclasses.replace`.
- Why no service-layer changes were needed: the assembled catalog already flows through `UnifiedMCPControlPlaneService.effective_tool_states()` (Tools State column + Permissions matrix rows) and `set_tool_state()` (matrix cycle writes, definition-hash fingerprinted) with server keys opaque — so `local:__local__` rows resolve and persist through the same `mcp_permissions.json` the Console gates on. The existing permission cycle's `_tool_for()` catalog lookup finds the new rows, so "allow" writes get their rug-pull hash (`local:__local__` is deliberately NOT in `HASH_FREE_SERVER_KEYS`).
- `executable=False` is set at the workbench layer, not in the provider's view (which stays invocation-capable): `mcp_inspector._test_gate_state()` renders the honest "not_executable" state from it, and no `tools/call` path to the local provider was opened. Hub-side execution is the deliberate follow-up.
- Gating (found by regression): an always-on local group put `mutates`-tagged rows into every fake-service permission-matrix test — the matrix correctly grew its Tags column and a Local workspace section, breaking 67 existing expectations. The group is therefore gated on `[console] local_tools_enabled` (coerced read), the same master opt-in the Console composition (`_compose_local_provider`) and the external MCP exposure (`[mcp] expose_local_tools`) already apply to this workspace-writing tool set; all 220 workbench tests pass unchanged, with three new tests pinning flag-on listing, flag-off absence, and fail-soft degradation.
- Fail-soft: ANY failure in provider construction/root resolution logs a warning and yields no local group — the profile/built-in catalog is never broken or emptied (pinned by a monkeypatched-constructor test).
- Modified: `tldw_chatbook/Agents/local_tool_provider.py`, `tldw_chatbook/UI/MCP_Modules/mcp_workbench.py`. Added tests: `Tests/Agents/test_local_tool_provider.py` (3 hub_tools cases), `Tests/UI/test_mcp_workbench.py` (catalog group + fail-soft), `Tests/MCP/test_control_plane_permissions.py` (shared-store allow round-trip, mutates risk floor + explicit-allow exemption). The implementation shipped to `dev` in PR #1435 (`24edce0f3f`); its acceptance criteria are complete. TASK-3605 owns the deliberately deferred Hub execution path.

## Implementation Plan

ADR required: no
ADR path: N/A — governed by existing ADR-030 (local library agent tool boundary), ADR-032 (local tool naming, workspace confinement, approval discipline) and the task-201 MCP Hub design
Reason: read-only surfacing of an existing catalog through existing seams (`LocalToolProvider.hub_tool_for`, the shared `mcp_permissions.json` store the Hub already reads/writes via `effective_tool_states`/`set_tool_state`). No storage/schema change, no new security boundary, no execution path opened — the Console and external-client gates are untouched.

1. Add `hub_tools()` to `Agents/local_tool_provider.py`: registration-order list of the provider's own `hub_tool_for()` views (already the exact payload the permission store fingerprints).
2. `UI/MCP_Modules/mcp_workbench.py`: new `_local_agent_hub_tools()` fail-soft helper — gated on `[console] local_tools_enabled` (coerced; the same opt-in the Console composition and `[mcp] expose_local_tools` apply to this workspace-writing tool set), builds a catalog-view `LocalToolProvider` (workspace root via `MCP/local_server_tools.resolve_server_workspace_root()`; no todo_store so `todo_write` stays absent; no approval callbacks — states resolve hub-side), maps each view to `executable=False` (`dataclasses.replace`) until Hub-side execution is wired; returns `[]` + warning log on any failure. Extend `_collect_hub_tools()`'s local-source branch to append it.
3. No service-layer changes: the assembled catalog already flows through `effective_tool_states()` (Tools State column + Permissions matrix) and `set_tool_state()` (matrix writes) with server keys opaque — verify with tests, including a `local:__local__` tool-level allow/ask/deny round-trip and the `mutates`-tag risk floor.
4. Tests: `Tests/Agents/test_local_tool_provider.py` (`hub_tools()` contents/order/todo_write absence), `Tests/UI/test_mcp_workbench.py` (catalog inclusion under local source, `executable=False`, fail-soft on provider-construction raise), permission round-trip coverage in the appropriate existing MCP permission test module.
5. File the follow-up task for Hub-side execution (Test Tool panel routing through a fail-closed provider, mirroring `local_server_tools.build_server_local_provider`).
