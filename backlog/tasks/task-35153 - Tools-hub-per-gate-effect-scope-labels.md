---
id: TASK-35153
title: 'Tools hub: per-gate effect-scope labels'
status: Done
created_date: 2026-10-16 20:15
updated_date: 2026-10-21 05:08
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Spec amendment A7 item 1 (Docs/superpowers/specs/2026-10-15-tools-hub-in-settings-design.md): most gates apply to the next chat (Console rebuilds its tool catalog per run); web_deep_search_enabled needs a restart (external MCP server). Add one effect-scope field to the all_tool_gates() table (Agents/builtin_tool_gate.py) so the hub, the Settings category, and the first-run Tools step render the same truth instead of three copies of it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 all_tool_gates() rows carry an effect-scope value,Hub/Settings/first-run render from the shared field,Tests cover next-chat vs restart scopes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Verify each gate's true effect scope from code: builtin rows + master + ask_user + character gates are read per Console run (next chat); web_deep_search is also read once at MCP server startup for external publication (restart)\n2. Add effect_scope field (next-chat|restart) + one short human label per scope + suffix helper to Agents/builtin_tool_gate.py; derive restart_required from it (single truth); scope assignment table consumed by all_tool_gates() rows\n3. Render the per-gate suffix from the shared field in the gate Button labels (mcp_servers_mode.py - serves the hub pane and Settings Tools & MCP) and in the first-run Tools step rows\n4. RED tests first (unit: scopes on rows + suffix rendering; UI: hub pane labels, Settings panel, wizard rows), then GREEN; run adjacent existing tests\n5. Update task notes + Done
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- **Approach.** One `effect_scope` field (`"next-chat"` | `"restart"`) on the `ToolGate` row, assigned from ONE config-free table `gate_effect_scopes()` that `all_tool_gates()` rows read from (a new gate without a scope fails loudly with KeyError instead of shipping an unlabeled row). `EFFECT_SCOPE_LABELS` holds one short human string per scope (`"next chat"` / `"restart to apply"`); `tool_gate_effect_suffix()` is its only rendering (`" (next chat)"` / `" (restart to apply)"`). The pre-existing `restart_required` boolean became a DERIVED property of the field (its only consumer, the pane's restart note, is unchanged) so the two can never drift.
- **Scopes assigned from code, not guesses.** `next-chat`: every `_GATEABLE_BUILTINS` row (read in `BuiltinToolProvider.__init__`, built per Console run), `[console] local_tools_enabled` (read per run in `_compose_local_provider`), `ask_user_enabled` (CONSOLE_ONLY, needs the Console-supplied callback — never consulted on the server path), `character_tools_enabled` (external publication is governed by the separate `[mcp] expose_character_tools` switch; `_character_specs` ships "no config gate applied" for that path). `restart`: `web_deep_search_enabled` only — the one `[tools]` key also read by `MCP/local_server_tools.py::build_server_local_provider()`, which the external MCP stdio server runs once at startup (`MCP/server.py`).
- **Surfaces rendered.** (1) The gate Buttons in `mcp_servers_mode.py`'s Tool gates pane (label suffix at all three `_tool_gate_label` call sites: build, optimistic press repaint, master projection) — this one widget serves both the hub pane and the Settings ▸ Tools & MCP category (the category embeds `MCPWorkbench`; there is no second rendering path). (2) The first-run Tools step's row-name Labels (`first_run_tools_step.py`), joining the shared scope table by gate_key (config-free by design — wizard gate state stays sourced from the in-memory `app_config` prefill, never a disk read). Zero new CSS (label text only, existing styles).
- **Behavior-neutral.** No gating logic, config reads, or write paths changed; the field only adds label text. The pane's two apply-notes (next-run note, web_deep_search external-publication note) are unchanged — the suffix states the per-row half, the notes keep the why.
- **Test seam fix (incidental, required to verify).** Three sibling tests in `Tests/UI/test_mcp_servers_mode.py` still patched `get_cli_setting` although task-32804.7 rerouted `all_tool_gates()` through one `load_cli_config_and_ensure_existence()` load — 68dc7d0e14 migrated only the workbench siblings, and these three kept passing against the local machine's real config by coincidence. Migrated them to the loader seam while updating their label assertions (exact-label asserts in `test_mcp_workbench.py` and the wizard's shared-copy test updated for the suffix). Recorded as a lesson in `backlog/docs/lessons-testing-evidence.md`.
- **Tests.** Unit (`Tests/Agents/test_builtin_tool_gate.py`): every row carries a valid scope; restart pinned to `web_deep_search_enabled` alone; representative next-chat gates per kind; `restart_required` derived-equivalence; suffix strings; the scope table is config-free and byte-identical to the rows. Rendering: servers-mode pane per-gate suffix + optimistic-repaint keeps suffix; Settings ▸ Tools & MCP category (private-profile, drives the real pane to the built-in detail) shows both scopes; first-run step names carry the next-chat suffix from the shared table. RED first, then GREEN.
- **Verification.** 153 passed across `test_builtin_tool_gate.py`, `test_mcp_servers_mode.py`, `test_mcp_gate_label_layout.py` (paint coverage at 80x24/170x48 with the longer labels), the 100x30 master-switch reachability test, and the new Settings-category test. Adjacent suites (`test_mcp_workbench.py`, `test_mcp_master_settings.py`, `test_settings_tools_hub.py`, wizard suites, `test_character_local_tools.py`, `test_ask_user_tool.py`, `test_document_expansion_tool.py`, `test_console_local_review_hook.py`, ratchet, token governance): failures byte-identical to a `git stash` baseline of the same node IDs on the pre-change tree (pre-existing env blocks: `RecoveryRequired` + stale dev ratchet rows) — zero regressions.
- **ADR required: no.** Copy/label addition over an existing enumerator; no storage, sync, provider-boundary, security, or interface decision. Renders an existing truth (already encoded as `restart_required`) through one shared field per TASK-35018/35019 copy rules.
- **Modified files.** `tldw_chatbook/Agents/builtin_tool_gate.py`, `tldw_chatbook/UI/MCP_Modules/mcp_servers_mode.py`, `tldw_chatbook/UI/Wizards/first_run_tools_step.py`, `Tests/Agents/test_builtin_tool_gate.py`, `Tests/UI/test_mcp_servers_mode.py`, `Tests/UI/test_mcp_workbench.py`, `Tests/UI/test_settings_tools_hub.py`, `Tests/Wizards/test_first_run_setup_wizard.py`, `Docs/User_Guide/mcp.md`, `backlog/docs/lessons-testing-evidence.md`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
One `effect_scope` field on the `all_tool_gates()` rows (`next-chat` for every gate except `web_deep_search_enabled` = `restart`), rendered as a per-row suffix ("(next chat)" / "(restart to apply)") from that one field in the gate Buttons (hub pane + Settings ▸ Tools & MCP, one widget) and the first-run Tools step rows; `restart_required` is now derived from it. Behavior-neutral; 153 targeted tests green; adjacent failures identical to the pre-change baseline.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
