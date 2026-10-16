---
id: TASK-35150
title: Retire MCP destination into Settings Tools & MCP hub
status: Done
created_date: 2026-10-16 16:30
updated_date: 2026-10-16 20:16
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The MCP destination is protocol jargon and a whole top-level destination for tool config. Fold all four workbench modes into one Settings category per the approved spec.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Settings has a Tools & MCP category hosting the four-mode MCPWorkbench
- [x] #2 mcp/tools_settings routes alias to Settings with category context
- [x] #3 ctrl+9 preserved as compatibility deep-link
- [x] #4 mcp-defaults read-only category deleted
- [x] #5 Tool Profiles handoff lands on the pane in Permissions mode
- [x] #6 targeted tests green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Pane module + TOOLS category registration (destination still live)
2. Keybindings a/r/t/1-4 + per-mode footer hints
3. Deep links + Tool Profiles handoff
4. Destination retirement + route aliases + compat hotkey
5. Test migration sweep
6. Docs + verification
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented per ADR-237; see Implementation Notes below.

**Approach.** Re-hosted the existing four-mode `MCPWorkbench` intact inside a
new sibling pane module (`UI/Screens/settings_tools_hub.py`) registered as
Settings category `SettingsCategoryId.TOOLS` ("Tools & MCP"); the destination
retirement follows the ADR-172 template (route aliases `mcp`/`tools_settings`
→ `settings` in both `_ROUTE_MAP`/`_ROUTABLE_LEGACY_ROUTES` and
`screen_registry._SCREEN_ALIASES`, default nav context `{"category": "tools"}`,
`MCP_COMPATIBILITY_SHORTCUT = "ctrl+9"` as an app-level hidden binding, palette
alias terms on the Settings command). Keys dispatch per-category: `1-4` are new
self-guarded mode bindings, `a`/`r`/`t` gain a TOOLS branch while keeping their
old meaning in every other category; per-mode footer hints ride the category
footer-shortcuts path. Spec:
`Docs/superpowers/specs/2026-10-15-tools-hub-in-settings-design.md`; ADR:
`backlog/decisions/237-mcp-destination-retirement-tools-hub-in-settings.md`
(renumbered from 233 — numbers 233-236 are claimed on sibling branches).

**Key decisions.**
- Footer registration uses a last-writer-wins tail-call: the pane re-asserts
  the category-aware registration after mode changes so hints stay per-mode
  honest without fighting the screen-level registrar.
- Recovery-review receipt invalidation is now production-owned on the
  workbench (`MCPWorkbench` exposes the invalidate hook; the Settings host
  calls it from its own suspend path) instead of living in the retired
  screen's `on_screen_suspend`.
- No-draft honesty for TOOLS: none of the four modes advertises a save
  affordance or Ctrl-S hint, and the page-save path excludes the category
  (workbench is immediate-write).
- Narrow-viewport regression accepted and filed: the workbench grid needs
  ~66 columns and Settings' detail pane gives ~terminal−41, so the category
  needs ~150 columns where the old destination worked at 80 — follow-up
  TASK-35151; further follow-ups filed as TASK-35152 (palette duplicate
  Switch-to-Settings entry), TASK-35153-35155 (spec A7 UX passes), TASK-35156
  (deep-link context merge for legacy aliases, spec A4 contract note).
- Startup configs with `default_tab = "mcp"` land on Settings → Tools & MCP
  via the alias + default-context map.

**Modified files.** New pane module `UI/Screens/settings_tools_hub.py`;
`UI/Screens/settings_screen.py` (category registration, key dispatch,
footer/category glue, deep-link context branch, runtime-backend forwarding,
suspend invalidation); `UI/Screens/mcp_screen.py` deleted;
`UI/Navigation/shell_destinations.py` + `UI/Navigation/screen_registry.py` +
`app.py` (retirement seams: order/shortcuts, aliases, compat binding,
default-context entries); `Constants.py` needed no change on this branch
(`TAB_MCP`/its label are untouched and now serve purely as compatibility
references); ~18 test files ported (destination shells, navigation aliases,
footer hints, permissions handoff, retired-destination root state, recovery
review, visual parity, smoke/perf guards).

**Ratchet measurement (spec A2).** `settings_screen.py` grew +211 lines on
this branch (36,444 → 36,655; 36,641 before the final-review view-state
fix) and is exempt from the module-size ratchet by
design (no row yet — the #3057 pin never landed first). The 12 red ratchet
modules were already red at the pristine merge base — line-count-proven on
untouched files (app_lifecycle 2337 > 2134, app.py 5824 > 5712,
mcp_workbench 6878 > 6760) — so no offset extraction was warranted; the
`mcp-defaults` deletion offset stands on its own.

**Category-count note (spec A1).** Category-count neutral — `MCP_DEFAULTS`
out, `TOOLS` in, enum stays 33 — so TASK-35020's rail-size guard can be pinned
once at 33; TASK-35016's "Owned elsewhere" group shrinks by one.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
