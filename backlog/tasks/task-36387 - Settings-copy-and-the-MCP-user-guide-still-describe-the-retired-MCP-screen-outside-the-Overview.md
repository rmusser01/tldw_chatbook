---
id: TASK-36387
title: Settings copy and the MCP user guide still describe the retired MCP screen outside the Overview
status: Done
assignee:
- '@claude'
created_date: 2026-10-22 23:58
updated_date: 2026-10-23 16:10
labels:
- follow-up
- settings
- copy
- docs
priority: low
references:
- tldw_chatbook/UI/Screens/settings_screen.py
- tldw_chatbook/UI/Screens/settings_tools_hub.py
- Docs/User_Guide/mcp.md
- Docs/User_Guide/index.md
dependencies:
- TASK-35647
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the implementation and review of TASK-35647 (PR #3218, 2026-10-22). Once TASK-35647 lands, the Settings Overview and the Tools & MCP banner name Settings ▸ Tools & MCP. Other copy still describes the layout before task-35150 (#3140), which retired the MCP destination:
- **Tool Profiles copy.** Its scope text says "MCP owns policy editing" (settings_screen.py around :2501 and :10853).
- **The ownership record.** `TOOL_PROFILES_OWNERSHIP_RECORD` (settings_tools_hub.py :103-117) says "MCP Permissions owns policy editing".
- **The User Guide.** `Docs/User_Guide/mcp.md` and `index.md` describe a standalone MCP screen ("Ctrl+9 … ⌃9 MCP in the nav bar"). Ctrl+9 itself is kept on purpose: `MCP_COMPATIBILITY_SHORTCUT` (shell_destinations.py :216) deep-links to Settings ▸ Tools & MCP.
- **Badges.** Most category-list statuses are not save models: "Guided", "Local", "Console", "Global gate" and so on. The State badges are save models, such as "Draft — save with s" and "Auto-saved". For the categories whose list status is a save model, the two can still name different ones. Workspaces, Dreams and Guardian list "Immediate actions" but badge "Applies immediately", and Tool Profiles badges "Reviewed actions". At dev 165e31fab8, the list statuses that say how edits save are "Immediate actions" and "Read-only" (Artifacts, Personas, Skills, Watchlists, Workflows and ACP Defaults, whose badge is "Read-only here"). TASK-35647 fixed this for Tools & MCP by having the badge read the category summary's status.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 No Settings copy or User Guide page describes MCP as its own screen; Ctrl+9 is documented as opening Settings ▸ Tools & MCP
- [x] #2 For each category whose list status says how its edits save ('Immediate actions' or 'Read-only' at dev 165e31fab8), its State badge says the same, from one definition
- [x] #3 A test fails if such a category's State badge and list status name different save models
<!-- AC:END -->

## Implementation Plan

1. One definition for the save-model list statuses: a `LISTED_SAVE_MODELS` table in `settings_config_models.py` (category -> "Immediate actions" / "Read-only", the statuses re-checked at 52b3207ff1). The category summaries (and `TOOLS_HUB_CATEGORY_SUMMARY`) take their status from it, and `_persistence_badge` returns it for those categories, replacing the Workspaces / Tool Profiles / Tools / Dreams / Guardian branches. A table rather than a `_category_summaries()` lookup, because that tuple reads config (Internal Prompts' count) and the badge is read by plain tests.
2. Settings copy: Tool Profiles' Boundary row, scope text and guided hint, its Edit policy tooltip, and `TOOL_PROFILES_OWNERSHIP_RECORD` name Tools & MCP instead of "MCP" / "MCP Permissions"; the hint leads with the badge's words.
3. User Guide: mcp.md's Getting there / first steps (Ctrl+9 opens Settings ▸ Tools & MCP; no nav label), index.md's screen table, keys note and legacy table, settings.md's badge table, category map and Domain Defaults table (MCP Defaults row is gone), and the other pages' own prose that names an "MCP screen".
4. Test in `Tests/UI/test_settings_tools_hub.py`: every category whose list status names a save model (a list save-model word or any badge) has a badge equal to it; non-vacuous over the eleven categories. Red at dev, green after.
5. Update pins of changed strings; keep settings_screen.py at or under its ratchet row; run covering tests, ratchets, preflight.

## Implementation Notes

**Approach.** The statuses at 52b3207ff1 match the description: "Immediate actions" for Workspaces, Tool Profiles, Tools & MCP, Dreams and Guardian; "Read-only" for Artifacts, Roleplay, Skills, Watchlists, Workflows and ACP Defaults. They now live in one table, `LISTED_SAVE_MODELS` (`settings_config_models.py`). The category summaries and `TOOLS_HUB_CATEGORY_SUMMARY` take their status from it. `_persistence_badge` returns it for those categories, which replaces five branches. The badge does not look the summary up through `_category_summaries()`, because that tuple reads config (the Internal Prompts count), and many plain tests read the badge.

**Badges that changed.** Workspaces, Dreams and Guardian ("Applies immediately") and Tool Profiles ("Reviewed actions") now read "Immediate actions". The six view-only Domain Defaults pages now read "Read-only". Overview, Diagnostics and About keep "Read-only here", because their list status names no save model.

**Copy.** Tool Profiles' Boundary row, scope text, guided hint (which now leads with "Immediate actions"), Edit tooltip and `TOOL_PROFILES_OWNERSHIP_RECORD` name Tools & MCP. The State line stays under 120 cells (119).

**User Guide.**
- mcp.md: Getting there now says Ctrl+9 opens Settings ▸ Tools & MCP and has no nav label; the palette entry is "Switch to Tools & MCP".
- index.md: the Ctrl+9 row, the global keys row, and the legacy-names table (which gains an MCP row).
- settings.md: the opening scope, the badge table (now nine badges, including the new Read-only row), the category map, and Domain Defaults. Domain Defaults is now six view-only pages: MCP Defaults is retired, and Schedules is not view-only.
- The prose in First_Run_Setup.md and console/agent-runs-and-tools.md that sent readers to "the MCP screen" or "MCP ▸ …".
- Verbatim quotes of non-Settings UI copy that still say "MCP ▸" are kept as quoted, with a note that this MCP is Settings ▸ Tools & MCP: `TOOL_GATES_PANE_PATH` (wizard summary and Permissions legend) and the approval card's scope lines.

**Tests.** In `Tests/UI/test_settings_tools_hub.py`:
- `test_state_badge_names_the_save_model_the_category_list_names` covers AC#3. A list status names a save model when it is "Immediate actions", "Read-only" or any badge's words. The test asserts at least the eleven categories, then compares badge to status. Red at dev source: a dict diff, Workspaces 'Applies immediately' != 'Immediate actions' (and Tool Profiles, Guardian, Artifacts, …).
- `test_tool_profiles_copy_sends_policy_editing_to_tools_and_mcp` is red at dev source on "Imports stay unbound; MCP owns policy editing; …".
- Both are green after the change.

**Pins rewritten on purpose.** Workspaces and Artifacts in `test_state_banner_leads_with_persistence_badge`, and Workspaces in `test_categories_without_a_draft_keep_todays_badge`. The plain hub tests were run under a temporary `@private_profile_test` marker, then reverted (the RecoveryRequired trap): 3 passed.

**Pre-existing reds, unrelated (not caused here).**
- `test_settings_tool_profiles.py::test_removal_confirmation_does_not_interpolate_the_profile_id`: also red with dev's source files.
- Four `Tests/Docs/test_console_library_controls_docs.py` cases on README.md and library/media-and-conversations.md, files this task does not touch.

**Ratchet.** settings_screen.py is 35,663 -> 35,660; its row is lowered.

**Files.**
- `tldw_chatbook/UI/Screens/settings_config_models.py`, `settings_screen.py`, `settings_tools_hub.py`
- `tldw_chatbook/Widgets/Settings_Widgets/tool_profiles_panel.py`
- `Docs/User_Guide/{mcp,index,settings,First_Run_Setup}.md`, `Docs/User_Guide/console/agent-runs-and-tools.md`
- `Tests/UI/test_settings_tools_hub.py`, `test_settings_configuration_hub.py`, `test_settings_state_line_unsaved_count.py`
- `Tests/Architecture/test_module_size_ratchet.py`
