---
id: TASK-40329
title: "The README's Application destinations table sends 9 of 14 keys to the wrong screen and leaves out Meetings, Plugins and Kanban"
status: Done
assignee: ['@claude']
updated_date: '2026-11-13 02:05'
created_date: 2026-11-12 19:36
labels:
- follow-up
- docs
- navigation
priority: medium
references:
- README.md
- tldw_chatbook/UI/Navigation/shell_destinations.py
- tldw_chatbook/UI/Navigation/screen_registry.py
- Tests/Docs/test_user_guide_contract.py
dependencies:
- TASK-38969
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (docs stream, TASK-38969). Checked by reading the code at the round-13 stack top 8b633d0101, 2026-11-12; line numbers are that tree's.

**Why.** The README is the first page people read. `README.md:449-462` (from d01daac8b4, 2026-08-30) no longer matches the destination map (`shell_destinations.py:210-226`, compatibility keys at `:231` Ctrl+6 and `:235` Ctrl+9):
- Ctrl+4 is Roleplay (README: Artifacts, retired by ADR-172; Ctrl+6 is kept as a compatibility key that lands in Library, `screen_registry.py:253`); Ctrl+5 is Watchlists (README: Roleplay); Ctrl+6 is the Artifacts compatibility key (README: Watchlists).
- Ctrl+9 is a compatibility deep link to Settings > Tools & MCP (ADR-240, `screen_registry.py:326`).
- Research is F5 (README: F10); Lab, Logs and Settings are F2, F3 and F4 (README: F7, F8, F9). F7 Meetings, F8 Plugins and F9 Kanban are missing.
- `README.md:655` says "Open **Logs** with **F8**"; Logs is F3 and F8 is Plugins.

TASK-38969's F-key drift test scans only the User Guide (`Tests/Docs/test_user_guide_contract.py:90`, `:120`), so it cannot catch the README.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The README table lists every destination shortcut with its destination, plus the Ctrl+6 and Ctrl+9 compatibility keys and where they land
- [x] #2 Every other hotkey mention in the README matches the map, including the Logs line
- [x] #3 A Docs test compares the README table with the destination map and is red on the current README
<!-- AC:END -->

## Implementation Plan

1. Re-check the destination map (`SHELL_DESTINATION_SHORTCUTS`, the Ctrl+6/Ctrl+9 compatibility bindings and where their actions land) at base 26b2e677dd.
2. Add a test to `Tests/Docs/test_user_guide_contract.py` that parses the README's "Application destinations" table and compares it with the shipped map (every destination key names its label; Ctrl+6/Ctrl+9 name the destination their route resolves to); show it red on the current README.
3. Rewrite the README table in key-walk order with all 15 destination keys plus the two compatibility rows; fix the "Open Logs with F8" line; check every other hotkey mention.
4. Run the Docs tests through the wrapper; commit.

## Implementation Notes

Rewrote the README's "Application destinations" table from the shipped map and fixed the one wrong prose key; a Docs test now compares both with the map.

**Re-checked at base 26b2e677dd** (all claims still held; line numbers are this tree's):
- `shell_destinations.py:210-227` `SHELL_DESTINATION_SHORTCUTS`: Ctrl+1 Home, Ctrl+2 Console, Ctrl+3 Library, Ctrl+4 Roleplay (`personas`), Ctrl+5 Watchlists, Ctrl+7 Schedules, Ctrl+8 Workflows, Ctrl+0 ACP, F2 Lab, F3 Logs, F4 Settings, F5 Research, F7 Meetings, F8 Plugins, F9 Kanban (15 keys).
- Ctrl+6 `:231` -> `app_lifecycle.py:1696` navigates to `artifacts`, which resolves to Library (`shell_destinations.py:306`) with mode `artifacts-all` (`:328`). Ctrl+9 `:235` -> `app_lifecycle.py:1700` navigates to `mcp` -> Settings (`:311`) category `tools` (`:335`).
- F1 Help / F6 Next Pane: `app.py:967-968`. Nav bar renders every `SHELL_DESTINATION_ORDER` entry (`main_navigation.py:481`); the palette lists every destination (`app_command_providers.py:301`) plus "Library — Artifacts" and "Switch to Tools & MCP" (`:324-339`).
- New rows' purpose text is the destination's own `purpose` (`shell_destinations.py:165-188`), shortened for Kanban.

**Changes.** `README.md`: table in key-walk order (15 destination keys + Ctrl+6 "Library ▸ Artifacts" + Ctrl+9 "Settings ▸ Tools & MCP"), a short note on F1/F6 and the nav bar/palette, and "Open **Logs** with **F3**" (was F8). The other hotkey mentions (Ctrl+2 Console, F1, Ctrl+P) already matched.

**Tests** (`Tests/Docs/test_user_guide_contract.py`, already on the derived-artifacts Docs lane): `test_readme_destination_table_matches_the_shell_shortcut_map` (each key's cell names the destination's label first; the compatibility keys name the destination `resolve_shell_route` gives their route) and `test_readme_prose_hotkeys_match_the_shell_shortcut_map` (every "**X** with **KEY**" sentence). Red on the old README: the table test listed 11 wrong/missing keys plus extra `f10`; the prose test failed on `('Logs', 'F8')`. Green after the edit: `Tests/Docs/` 32 passed.
