---
id: TASK-17652
title: 'Console: status chips placement setting (above/below composer) + persistent collapse'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-17'
labels:
  - console
  - ux
  - settings
dependencies:
  - task-17650
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The status-chip strip sits below the composer on dev (TASK-15704 moved it there and pinned the order). The owner wants a user-facing setting to place it either above or below the composer input. Default ruling updated 2026-08-17 after viewing the merged TASK-17650 live: the owner wants the status bar ON TOP of the composer row, so the shipped default is "above"; "below" remains available through the setting.

The 2026-08-17 audit mapped what "above" must respect: the prompt-queue shelf is pinned immediately above the composer (`queue.y + queue.h == composer.y`), so chips-above means directly under the workspace grid, ABOVE the staged-evidence/prompt-queue cluster — not wedged between the shelf and the composer. `ConsoleCommandPopup.reposition`'s clearance loop deliberately excludes the chips (they are "below, out of reach") and would paint over them on every `/` in above mode. The F6/Tab region pairing maps the chips to the transcript surface and must follow the visual position. Two currently-green contract tests hard-assert chips-below and need parameterizing over both modes.

Also in scope: the Status collapse state (`_console_status_chips_collapsed`) is session-only screen state today — it resets every time the user leaves and re-enters Console. Since this task adds `[console]` persistence plumbing anyway, persist the collapse state alongside the position.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 A `[console] status_chips_position` config key ("below"|"above", default "above" per the 2026-08-17 owner ruling) exists with a staged Settings > Console Behavior control using the 1-row toggle pattern, indexed for field search
- [x] #2 In above mode the chips render directly under the workspace grid, above the staged-evidence/prompt-queue cluster; the prompt-queue shelf stays immediately adjacent to the composer in both modes
- [x] #3 The command popup never paints over the chips in either mode
- [x] #4 F6/Tab region pairing stays coherent in both modes (the chips keep their transcript-region pairing from CONSOLE_TAB_REGIONS — physically adjacent in above mode, and identical to shipped dev behavior in below mode; tab-scope suite green in both)
- [x] #5 The order and popup contract tests are parameterized over both positions and green
- [x] #6 The Status collapse state persists across Console re-entry and app restart
- [x] #7 User Guide Console and Settings pages updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: flip the order contract — `test_console_status_chips_sit_below_composer` becomes the above-by-default contract (chips directly under the workspace grid, above the staged/queue/composer cluster) plus a below-mode variant driven by `[console] status_chips_position = "below"`; same for the popup anchor test (popup bottom clears the chips in above mode).
2. New module `UI/Console_Modules/status_row.py` (ratchet: chat_screen.py is already over budget — new logic lives in Console_Modules): position/collapse resolvers with validation, `apply_status_chips_position(screen)` DOM mover, never-raising persist helpers.
3. chat_screen.py, thin delta only: compose builds the chips once and yields them above or below per the resolver; `__init__` seeds `_console_status_chips_collapsed` from config; the collapse setter pokes the in-memory config and persists via `run_worker(thread=True)`; `on_screen_resume` calls `apply_status_chips_position` so a Settings change takes effect on return without recompose (ChatScreen is cached across navigation).
4. `ConsoleCommandPopup.reposition`: add `#console-status-chips` to the clearance loop — `min()` semantics make it inert in below mode (chips y is larger than the anchor) and correct in above mode; update the DS-09 comment.
5. Settings ▸ Console Behavior: "Status row placement" section following the remote-images ADR-020 immediate-toggle pattern (button label carries the state, threaded persist, in-memory poke; copy says it takes effect on returning to Console); FIELD_SEARCH_INDEX entries.
6. config.py: normalize `status_chips_position` (above|below, default above) and `status_chips_collapsed` (bool, default false) in the `[console]` block.
7. GREEN + full bottom-stack contract sweep; live headless row map in both modes; Docs (console.md layout tour currently says the strip is "directly below the composer"; settings page) + stamps.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Shipped exactly per the plan; one clarification on AC#1: the control uses the remote-images ADR-020 immediate-write pattern rather than the staged draft, matching the card's existing precedent for single self-contained toggles — the button label carries the state ("Above composer"/"Below composer"), presses persist off-loop and poke the live config, and the cached Console re-applies the position on `on_screen_resume` (no restart, no recompose).

New `UI/Console_Modules/status_row.py` holds all logic (ratchet: chat_screen.py is over budget, task-3751): position/collapse resolvers with validation, `poke_console_setting`, `apply_status_chips_position` (DOM `move_child` around the staged/queue/composer cluster), and a never-raising threaded persist. chat_screen.py got only thin deltas: compose builds the chips once and yields them on the configured side; `__init__` seeds the collapse state from config; the collapse setter pokes + persists; resume re-applies position. `ConsoleCommandPopup.reposition` adds the chips to its clearance loop — `min()` semantics make the entry inert in below mode (chips y exceeds the anchor) and correct in above mode, so the popup needs no setting-awareness.

TDD evidence: the order and popup above-mode contracts were watched RED first (chips still below), then GREEN; the two collapse-persistence pins were written after their wiring and therefore MUTATION-TESTED (init seed reverted → both RED; poke disabled → RED) before being trusted. The former `test_screen_status_collapse_state_resets_on_new_screen` pin was inverted to the persistence contract. One legitimate contract update: the rail-labels keyboard test pinned Shift+Tab adjacency between two checkboxes my toggle now sits between; the pin now documents the three-stop order.

Evidence: headless row maps at 150x44 in both placements (above: grid border y36, chips y37, composer y38-42, footer y43; below: composer y37-41, chips y42, footer y43; zero blank rows either way, transcript region h=29 in both); contract sweep 833 passed + the updated rail-labels file 5/5; ruff clean on all touched files (11 pre-existing findings at untouched lines).

Files: `tldw_chatbook/UI/Console_Modules/status_row.py` (new), `tldw_chatbook/UI/Screens/chat_screen.py`, `tldw_chatbook/UI/Screens/settings_screen.py`, `tldw_chatbook/Widgets/Console/console_command_popup.py`, `tldw_chatbook/config.py`, `Tests/UI/test_console_workbench_contract.py`, `Tests/UI/test_console_command_popup.py`, `Tests/UI/test_console_status_row_collapse.py`, `Tests/UI/test_settings_console_status_row.py` (new), `Tests/UI/test_settings_console_rail_labels.py`, `Docs/User_Guide/console.md`, `Docs/User_Guide/settings.md`.
<!-- SECTION:NOTES:END -->
