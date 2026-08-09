---
id: TASK-3222
title: Settings Tools screen does not show the web_deep_search gate
status: Done
assignee:
  - '@claude'
created_date: '2026-08-07 16:30'
updated_date: '2026-08-07 20:26'
labels:
  - web-tools
  - ux
  - settings
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
web_deep_search's enable switch is the TOML key [tools] web_deep_search_enabled (double opt-in per the task-1356 spec), but Settings ▸ Tools derives its rows from _GATEABLE_BUILTINS — a different registry — so the toggle never appears in the UI. This was a recorded spec non-goal (the tool ships config-file-only on purpose), but a user who finds other tool switches in Settings will reasonably conclude this one does not exist. The final whole-branch review (2026-08-07) recommended a follow-up task rather than silence. Scope question for whoever picks this up: surface the [tools] boolean as a read-only or editable row, or generalize the Settings source so config-gated tools appear alongside _GATEABLE_BUILTINS. Note the restart-to-apply semantics (provider builds specs at construction) must stay visible in whatever UI ships.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Settings ▸ Tools shows the web_deep_search opt-in state (at minimum read-only with the config key named)
- [x] #2 The restart-to-apply requirement is stated wherever the state is shown
- [x] #3 Toggling (if editable) round-trips to [tools] web_deep_search_enabled in config.toml
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Locate Settings ▸ Tools switch derivation (_GATEABLE_BUILTINS)\n2. Add a row for web_deep_search's [tools] web_deep_search_enabled gate (editable switch writing the same key, matching sibling rows) with restart-to-apply note\n3. Tests: row present, round-trips the key, restart note visible
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Shipped an editable switch row, not read-only: ToolsSettingsWindow's existing save/reset machinery (Save Tool Settings / Reset Tool Settings buttons, [tools] round trip via save_settings_to_cli_config) made it cheap, and the pre-existing controller plan called for editable.

web_deep_search (Agents/local_tool_provider.py, a LocalToolSpec) is NOT a builtin Tool ABC subclass, so it cannot join _GATEABLE_BUILTINS (Agents/tool_catalog.py) -- GateableTool requires a module/class to instantiate. Added it as a standalone row in Tools_Settings_Window.py's _compose_tool_settings, right after the gateable_builtin_tools() loop, using the same tool-switch-{name} id convention, Horizontal(classes="tool-item") structure, and [tools] table. _save_tool_settings and _reset_tool_settings each got a small explicit branch (query the extra switch by id, fold it into the same updates dict / reset-to-False pass) since it falls outside the gateable_builtin_tools() loop those methods iterate.

Two new module constants (WEB_DEEP_SEARCH_GATE_KEY = "web_deep_search_enabled", WEB_DEEP_SEARCH_TOOL_NAME = "web_deep_search") avoid hardcoding the string in five places. The row's description states the config key as dotted tools.web_deep_search_enabled (not bracketed [tools] ...) to avoid Rich/Textual markup parsing -- the file's own existing comment already flags this exact bracket-as-markup trap for risk tags. Restart-to-apply text: "restart the app after saving: the provider builds its tool list once at startup, so this switch has no effect on the current session" -- matches local_tool_provider.py's own _default_specs comment.

Architecture note worth flagging: Tools_Settings_Window.py carries a DEPRECATED (TASK-1346) banner ("Do not add new settings here... not reachable through normal navigation") -- the canonical settings surface is UI/Screens/settings_screen.py, whose MCP Defaults category deliberately stays read-only ("add MCP defaults only after server-first settings are exposed without flattening tools into Settings"). No _GATEABLE_BUILTINS switch is reachable through live navigation today; ToolsSettingsWindow is exercised only by its own direct-push test harness (Tests/UI/test_tools_settings_window.py). This task's own pre-approved Implementation Plan pointed at _GATEABLE_BUILTINS' UI (this file) explicitly, matching CLAUDE.md's "New Tool" guidance verbatim, so followed that rather than re-litigating the migration -- but a future cleanup of ToolsSettingsWindow must carry this row (and the whole Tool Settings section) forward, not drop it silently.

Found and fixed a test-authoring trap while writing the widget-level tests: with Horizontal(classes="tool-item"): yield Switch(...) inside _compose_tool_settings does NOT nest children under the Horizontal in the live tree -- *self._compose_tool_settings() is spliced as a pre-materialized tuple into the outer Container(...) call, so the generator (and every Horizontal it enters/exits) fully drains before that Container is ever mounted/entered. Every row's Switch/Label/Static end up flat siblings directly under Container(id="ts-view-tool-settings"), not nested -- true for every existing GateableTool row too, not just this one. Tests query the whole view and match by content instead of assuming a scoped Horizontal wrapper.

Files: tldw_chatbook/UI/Tools_Settings_Window.py (row + save/reset + constants); Tests/UI/test_tools_settings_window.py (5 widget-mount tests: row+restart-note present, default-off, reflects-enabled-config, save round-trip, reset-to-off); Tests/UI/test_settings_tools_section.py (2 tests: config-to-provider round trip via _default_specs, source-scan regression pin).

**Controller resolution (2026-08-07) of the architecture note above:** verified and real — the "tools_settings" route resolves to MCPScreen and the canonical settings_screen has no Tools category, so EVERY tool-gate switch (not just this row) is nav-unreachable after first run; the only live gate surface is the FirstRunSetupWizard. That product gap is bigger than this task and is filed as task-3240 (owner decision on fix shape; settings_screen.py is under active critique work on another branch, so the Tools-category option needs coordination). This task's row deliberately ships where the sibling machinery lives, so it appears the moment that surface is re-routed or carried forward.

**Correction (2026-08-07, owner-flagged):** the paragraph above overclaimed — "EVERY tool-gate switch is nav-unreachable" conflated two layers. The MCP screen's Tools/Permissions modes ARE a live, reachable tool-management surface (catalog browse incl. the local agent tool group; per-tool Allow/Ask/Off). What is unreachable is specifically the [tools] registration-gate SWITCHES (catalog membership) — and since a gate-off tool is absent from every catalog the MCP screen derives from, the gate can't be flipped from that surface either. task-3240's description carries the corrected scoping.

**Epilogue (2026-08-08, task-3240):** the "appears the moment that surface is re-routed or carried forward" line above did not happen that way — task-3240 shipped a NEW, independent affordance instead (Servers mode ▸ built-in detail ▸ "Tool gates" group, backed by a fresh `all_tool_gates()` enumerator). `web_deep_search`'s gate is now reachable and flippable there. `ToolsSettingsWindow`'s row (this task's own work) remains exactly where it was — still correct, still DEPRECATED, still nav-unreachable; the dead window remains dead.
<!-- SECTION:NOTES:END -->
