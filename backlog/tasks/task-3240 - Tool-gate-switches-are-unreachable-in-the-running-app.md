---
id: TASK-3240
title: Tool gate switches are unreachable in the running app
status: Done
assignee:
  - '@claude'
created_date: '2026-08-07 20:15'
updated_date: '2026-08-09 04:39'
labels:
  - settings
  - ux
  - tools
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**Corrected scope (2026-08-07, owner correction — the original filing overclaimed):** the original description said "no tool gate can be flipped from inside the app at all", ignoring the MCP screen. That was wrong about tool management generally: the MCP screen's Tools mode is a live, nav-reachable hub catalog (MCP servers + builtin inventory + the local agent tool group via _local_agent_hub_tools when [console] local_tools_enabled is on), and its Permissions mode sets Allow/Ask/Off per tool against mcp_permissions.json — including builtin and local tools. That is the per-call permission half of the double opt-in, and it works today.

The real gap is one layer narrower: the **[tools] <name>_enabled registration gates** — the config switches that decide whether a tool exists in a catalog at all (the other half of the double opt-in). Their only switch UIs are UI/Tools_Settings_Window.py (deprecated by TASK-1346; the "tools_settings" route resolves to MCPScreen) and the FirstRunSetupWizard (seen once). And because everything the MCP screen shows is downstream of the gates — builtin_permission_rows enumerates a BuiltinToolProvider, which only registers gate-enabled tools; a gate-off web_deep_search is absent from LocalToolProvider._default_specs and therefore from hub_tools — **a gate-off tool is invisible to the entire live surface**: it can't be discovered or enabled in-app after first run, only by hand-editing config.toml.

Fix-shape options (owner decision): (a) a gate affordance in the MCP hub — arguably the natural home, since it already lists local tools and manages the permission layer; e.g. show gate-off tools greyed with an explicit enable action; (b) a Tools category in the canonical settings_screen (under active critique work on fix/settings-ux-critique-rounds — coordinate); (c) re-route "tools_settings" to a live wrapper of the existing window. Restart-to-apply semantics for construction-time gates (web_deep_search) must stay visible in whatever ships.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A user can discover and flip [tools] registration gates (including gates currently OFF) through the running app's navigation after first run
- [x] #2 All [tools] gate switches (gateable builtins + web_deep_search) plus the [console] local_tools_enabled master switch are present there and round-trip to config.toml (amended 2026-08-08: the master switch declared as an intentional extension — it masters the local group the gates live in)
- [x] #3 Gates that need an app restart to apply state that where shown
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Premise-check on latest dev (done — valid; MCP hub redesign is COMPLETE not pending; greyed-matrix-rows sketch structurally wrong)\n2. Spec: gate checkboxes in Servers-mode source detail (builtin + local), unified all-gates enumerator, save path mirroring _save_builtin_flag, empty-state breadcrumbs\n3. Adversarial spec review, implement, opus review, PR
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Shipped the MCP-hub gate affordance per spec branch (b): a unified all_tool_gates() enumerator (Agents/builtin_tool_gate.py) plus a Tool gates checkbox group in Servers mode's builtin-source detail pane, saving via a new ToolGateChanged message and MCPWorkbench._save_tool_gate(). See Implementation Notes below for the full breakdown.

**Design:** `Docs/superpowers/specs/2026-08-08-mcp-gate-affordance-design.md` (adversarial-reviewed; branch (b) resolved the local-source detail question, since `_collect_snapshots()` never produces a `local:__local__` row).

**Order of work / commits (feat/mcp-gate-affordance-3240):**
1. `fix: coerce the builtin-tool registration gate read` — the Critical prerequisite. `BuiltinToolProvider.__init__` (Agents/tool_catalog.py) read `[tools] <x>_enabled` via raw `not get_cli_setting(...)`, the last un-coerced gate read in the codebase (the arc's fifth `bool("false")` site) — a quoted `"false"` is a non-empty string and therefore truthy, so it silently REGISTERED the tool while a coerced UI would show it OFF. Wrapped in `coerce_bool_setting`, TDD (regression tests written first, confirmed red, then fixed).
2. `feat: add a unified [tools]/[console] gate enumerator` — `ToolGate`/`all_tool_gates()`/`tool_gate_breadcrumb()` in Agents/builtin_tool_gate.py (beside the existing `builtin_permission_rows()` settings-time-enumeration precedent): 9 gates total — the 7 `_GATEABLE_BUILTINS` rows (group "builtin"), then the local group (group "local": `[console] local_tools_enabled` master switch listed first, then `web_deep_search`). Also relocated `WEB_DEEP_SEARCH_GATE_KEY` from UI/Tools_Settings_Window.py to Agents/local_tool_provider.py (its actual runtime consumer, which used to re-type the literal) — Tools_Settings_Window now imports it instead of defining it.
3. `feat: surface tool gate switches in the MCP hub's Servers mode` — the UI: a "Tool gates" checkbox group under two subheadings ("Agent built-ins" / "Local workspace tools") in `mcp_servers_mode.py`'s builtin-source detail, a new `MCPServersMode.ToolGateChanged(section, key, value)` message (unlike `BuiltinFlagChanged`, section is NOT hardcoded to "mcp" — task-3240's gates span both `[tools]` and `[console]`), routed through `MCPWorkbench._save_tool_gate()` (mirrors `_save_builtin_flag()`). Two discoverability breadcrumbs, both computed fresh from `tool_gate_breadcrumb()`: the Permissions matrix's always-visible legend (primary — `mcp_permissions_mode.update_matrix()`'s new `gate_breadcrumb` param, rendered as a second line under the fixed marker-key text) and `_empty_tools_diagnosis()` (secondary, honestly partial — only reachable when the Tools-mode catalog is fully empty).

**Docs:** CLAUDE.md's "New Tool" step 4 rewritten (gates now surface automatically via `all_tool_gates()`; a non-`_GATEABLE_BUILTINS` tool needs one hand-list entry there, the `web_deep_search` precedent). `Docs/User_Guide/mcp.md` gained a "Turning tools on (Servers mode ▸ Tool gates)" section and a refreshed "Verified against" stamp. task-3222's Implementation Notes got one epilogue line: superseded by this task's live affordance; its own `ToolsSettingsWindow` row remains correct but still dead/unreachable.

**Testing:** TDD throughout (failing test first, confirmed red, then the fix). New/updated: `Tests/Agents/test_tool_catalog.py` (2 coercion regression tests), `Tests/Agents/test_builtin_tool_gate.py` (7 enumerator/breadcrumb tests: 9-gate structure+order+groups, coerced-not-raw enabled reads x2, bool passthrough, constant-not-literal, breadcrumb present/absent), `Tests/UI/test_mcp_servers_mode.py` (4 new widget tests + 1 existing layout test updated for the new sibling container), `Tests/UI/test_mcp_permissions_mode.py` (2 new legend tests), `Tests/UI/test_mcp_workbench.py` (1 round-trip test pinning the `[tools]`-vs-`[console]` section threading, 2 new breadcrumb tests, 3 pre-existing empty-diagnosis tests loosened from `==` to `.startswith(...)` since they now legitimately carry a trailing breadcrumb). Seam-namespace trap (spec review Minor 6, confirmed by writing it): `all_tool_gates()`/`BuiltinToolProvider.__init__` read via a FUNCTION-LOCAL `from ..config import get_cli_setting`, so tests controlling their reads must patch `tldw_chatbook.config.get_cli_setting` directly — patching `mcp_workbench`'s own imported name (the existing `_save_builtin_flag` test's seam) does not reach it; the round-trip test patches both, backed by the same in-memory dict.

**Mutation checks (Edit-based restores, unique anchors, RED confirmed before restoring):**
1. Un-coerced the `tool_catalog.py` registration read back to raw `not get_cli_setting(...)` → both new coercion regression tests went red (one asserting "false" is NOT registered, since raw truthiness would register it).
2. Dropped `read_file_enabled` from `_GATEABLE_BUILTINS` → the enumerator's 9-gate count/order test went red.
3. Made `MCPServersMode.on_checkbox_changed`'s gate branch a silent no-op (restored the pre-task shape: only `_BUILTIN_CHECKBOX_KEYS` handled) → the round-trip test went red (no `ToolGateChanged` posted, no save call).
4. Hardcoded `tool_gate_breadcrumb()`'s off-count to a literal instead of `sum(...)` → the breadcrumb count tests went red.

All four restored via Edit (unique surrounding context, never `git checkout --`) immediately after confirming red.

**Gates before calling this done (all read as nonzero passed-counts, not just exit codes):** `Tests/Agents/test_tool_catalog.py` 22/22, `Tests/Agents/test_builtin_tool_gate.py` 31/31, `Tests/Agents/test_local_tool_provider.py` + `Tests/Tools/test_web_deep_search.py` 147/147, `Tests/UI/test_mcp_servers_mode.py` 56/56, `Tests/UI/test_mcp_permissions_mode.py` 56/56, `Tests/UI/test_mcp_workbench.py` 233/233, `Tests/UI/test_tools_settings_window.py` + `Tests/UI/test_settings_tools_section.py` 181/181 (the relocated-constant regression check).

**Files touched:** `tldw_chatbook/Agents/tool_catalog.py`, `tldw_chatbook/Agents/builtin_tool_gate.py`, `tldw_chatbook/Agents/local_tool_provider.py`, `tldw_chatbook/UI/Tools_Settings_Window.py`, `tldw_chatbook/UI/MCP_Modules/mcp_servers_mode.py`, `tldw_chatbook/UI/MCP_Modules/mcp_workbench.py`, `tldw_chatbook/UI/MCP_Modules/mcp_permissions_mode.py`, `CLAUDE.md`, `Docs/User_Guide/mcp.md`, plus the test files named above.

**Trade-off accepted (per spec, branch (b)):** the "Tool gates" group renders inside the pane badged as the built-in MCP *server* even though it controls the in-process *agent* tool catalog — a different subsystem. Mitigated with two subheadings and copy explicit about which subsystem is affected; a dedicated "Local workspace" rail row is out of scope for this task.
<!-- SECTION:NOTES:END -->
