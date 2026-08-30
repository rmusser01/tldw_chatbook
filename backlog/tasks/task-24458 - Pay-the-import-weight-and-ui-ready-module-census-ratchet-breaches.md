---
id: TASK-24458
title: Pay the import-weight and ui-ready module census ratchet breaches
status: Done
assignee: []
created_date: '2026-08-29'
labels:
  - performance
  - boot
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Both boot module ratchets are red on pristine dev:
`MAX_TLDW_MODULE_COUNT` is 662 against a limit of 660, and `MAX_TLDW_MODULES_AT_UI_READY` is
980 against a limit of 970.

TASK-23112 paid the import ratchet down to 646/660 (headroom 14) on 2026-08-28. It was breached
again within a day. The newly resident modules cluster in one family -- the workspace and
tool-execution work: `Tools.{git,local,patch,virtual_cli}_tool_impls`,
`Tools.workspace_tool_{executor,protocol}`, `Tools.workspace_root_pin`,
`Agents.{raw_shell,virtual_cli}_tool_provider`, `Workspaces.change_review_{consent,finalization}`,
`Chat.console_settings_*`, `TTS.{legacy_request_builder,text_processing}`.

Per ADR-097 the constants must not be raised; the cost must be deferred off the boot path or
shed elsewhere.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `test_app_import_own_module_count_stays_at_the_post_diet_size` passes on a pristine checkout
- [x] #2 `test_ui_ready_module_census_stays_at_the_pinned_size` passes on a pristine checkout
- [x] #3 Neither `MAX_TLDW_MODULE_COUNT` nor `MAX_TLDW_MODULES_AT_UI_READY` is raised
- [x] #4 Each deferral is re-measured after it is applied, because the import-parent tracer records only the first importer and its attribution is an upper bound
- [x] #5 The tool-execution features whose modules were deferred still work on first use
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
Both ratchets green with headroom, constants untouched:
- boot import weight **662 -> 631 / 660** (headroom 29)
- `_ui_ready` census **981 -> 966 / 970** (headroom 4)

Four deferrals, each traced with a first-importer tracer and RE-MEASURED after applying (the
ADR-097 lesson: the tracer records only the first importer, so its attribution is an upper
bound -- and it proved so here, three separate edges had to be cut before the cluster left).

1. `app.py` imported `UI.Tools_Settings_Window` solely to name
   `ToolsSettingsWindow.IngestUiStyleChanged` in an `@on(...)` decorator -- which needs the class
   at class-body time, so it could not just move into a function. That import dragged
   `Agents.local_tool_provider` -> `Tools.workspace_tool_executor` and 7 more modules onto boot,
   for a window that is DEPRECATED (TASK-1346) and nav-unreachable. The message moved to a new
   4-line `UI/tools_settings_messages.py`; the window re-exports it as a class attribute so
   `ToolsSettingsWindow.IngestUiStyleChanged` and `self.IngestUiStyleChanged(...)` are unchanged.
2. `console_chat_controller` imported `RawShellToolProvider`, `VirtualCliProvider` and the
   `RAW_SHELL_*` constants at module scope; moved to their runtime use sites. Safe because the
   module has `from __future__ import annotations`, so no type reference evaluates at runtime.
3. Same for `LocalToolProvider` in that controller.
4. The last edge was the subtle one: `console_agent_bridge` imported six refusal STRINGS from
   `local_tool_provider` to build a module-scope frozenset -- seven modules resident to compare a
   handful of strings. The set is now built by an `lru_cache`d `_blocked_provider_refusals()`.

### The two ratchets interact, and the second one caught it

Rebasing onto dev turned `test_screen_preimport_payload_budget` RED at 505/500 modules while
it PASSES on pristine dev. The module-set diff named exactly the 14 modules these deferrals
moved off the boot path.

That guard measures `tldw_modules()` AFTER the pre-import pass minus a baseline taken BEFORE
it. Reducing what boot imports therefore lowers the baseline and RAISES `pass_added` by the
same amount, with no new work done anywhere. Total modules loaded is unchanged; the accounting
moved. Dev had only 226 LOC of headroom, and this shift moved 842 LOC of accounting into the
census, so it breached.

Rather than raise a constant (ADR-097 forbids it) or revert a real boot win, the payload was
reduced by 653 LOC of genuinely wasted eager imports, all found by following the same edges:

- `UI/Widgets/__init__.py` eagerly imported `SmartContentTree` (425 LOC) and
  `config_search_widget` (228 LOC). The four MCP modes import only the small
  `table_click_select` mixin from that package, so every one of them paid for both. Now
  resolved lazily via PEP 562 `__getattr__`. **This is the `Chunking/__init__.py` trap
  (finding 21102) in a second package.**
- `UI/MCP_Modules/mcp_workbench.py` and `MCP/local_server_tools.py` imported the same
  workspace tool providers at module scope that `console_chat_controller` did; both are
  reached by the pre-importer. Deferred to their runtime use sites.
  `local_server_tools` has no `from __future__ import annotations`, so its three
  `LocalToolProvider` annotations are QUOTED rather than switching the whole module to
  string annotations.
- `settings_screen` imported `Utils.about_text` at module scope for one branch of one
  category; moved to where it renders.

Final on the first base: preimport payload 492/500 modules, 379,955/380,000 LOC (headroom 45).

**Then dev moved 66 commits and broke both guards on its own.** Measured on pristine dev
`bf23929857`: `_ui_ready` census **983/970 RED**, preimport payload **383,038/380,000 LOC RED**.
On this branch, rebased onto that same dev: census **970/970 GREEN** (this PR repairs a dev red)
and preimport **383,228** -- i.e. this branch contributes +190 LOC of the 3,228 breach, which is
the documented residual of the accounting shift above (842 moved, 653 shed). The preimport guard
cannot be made green from this branch and is dev's to repay.

**Two things the owner should know.** The census lands at exactly 970/970 with ZERO headroom, and
task-24461 wires it into the per-PR guard -- so the next PR that adds any module to the mount leg
fails fast. That is the ratchet working as designed, but it is a hair trigger. And the preimport
LOC guard runs in neither CI job (not the fast lane, not perf-guard), which is why dev drifted
3,038 LOC over it unnoticed.

Modified: `tldw_chatbook/UI/tools_settings_messages.py` (new), `tldw_chatbook/UI/Tools_Settings_Window.py`,
`tldw_chatbook/app.py`, `tldw_chatbook/Chat/console_chat_controller.py`,
`tldw_chatbook/Chat/console_agent_bridge.py`, `tldw_chatbook/UI/Widgets/__init__.py`,
`tldw_chatbook/UI/MCP_Modules/mcp_workbench.py`, `tldw_chatbook/MCP/local_server_tools.py`,
`tldw_chatbook/UI/Screens/settings_screen.py`.
<!-- SECTION:NOTES:END -->
