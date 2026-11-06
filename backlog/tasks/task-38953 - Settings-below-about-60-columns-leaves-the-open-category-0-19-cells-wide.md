---
id: TASK-38953
title: "Settings below about 60 columns leaves the open category 0-19 cells wide"
status: Done
assignee:
- '@claude'
created_date: 2026-11-05 21:35
updated_date: 2026-11-06 19:41
labels:
- follow-up
- settings
- ui
- responsive
priority: low
references:
- tldw_chatbook/UI/Screens/settings_screen.py
- tldw_chatbook/css/features/_settings.tcss
- Tests/UI/test_settings_category_sweep.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down; checked on dev 29592ecc9b, 2026-11-05.

At 100 columns and below, `SETTINGS_COMPACT_WORKBENCH_MAX_WIDTH = 100` (`tldw_chatbook/UI/Screens/settings_screen.py:972`) switches Settings to its compact two-pane layout. The category pane is fixed at 32 cells (`tldw_chatbook/css/features/_settings.tcss:117-121`), and the detail pane gets whatever is left, with `min-width: 0` (`:123-127`).

Measured in the bundled-CSS harness at height 30. Overview, Providers & Models and Tools all give the same numbers. Detail-pane content width by terminal width:

| Columns | 100 | 80 | 70 | 64 | 62 | 60 | 59 | 56 | 54 | 50 | 46 | 44 | 40 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Content cells | 59 | 39 | 29 | 23 | 21 | 19 | 18 | 15 | 13 | 9 | 5 | 3 | 0 |

The category pane keeps 32 cells (28 of content) the whole way down.

There is also a jump at the breakpoint. The three-pane layout gives the detail pane only 34 cells at 101 columns and 43 at 110, which is less than the compact layout gives at 80 columns (39).

Context:
- TASK-38740's sweep from 60 down to 40 columns checked only for crashes.
- Library already shows one pane at a time below 64 columns (TASK-32225, TASK-32360), so there is a precedent to follow.
- The app is normally used full-screen (211x44 or 235x52), hence the low priority.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 At every width from 60 down to 40 columns (heights 24 and 30), the open category gets at least 30 content cells, or Settings shows one pane at a time with a labelled way back to the category list; a test measures this and is red on dev (19 cells at 60 columns, 0 at 40)
- [x] #2 Every category stays reachable by keyboard at those widths, including the way back to the category list
- [x] #3 At 211x44 and 235x52 the Settings layout is unchanged
- [x] #4 The detail pane never gets narrower as the terminal gets wider across the 100/101-column breakpoint (on dev: 59 cells at 100 columns, 34 at 101), or the notes record why the jump stays
<!-- AC:END -->

## Implementation Plan

1. Measure the detail pane's content width against terminal width in the bundled-CSS harness, 140 down to 40 columns.
2. Tests first (red on dev): content width >= 30 from 60 to 40 columns at heights 24 and 30; a keyboard-only walk of every category with the way back; the pane widths at 211x44 and 235x52.
3. One pane at a time at or below the width where the compact rail leaves the open category under 30 cells, following Library's one-pane rule: classes on `#settings-workbench`, CSS in `_settings.tcss`, and a "‹ Categories" control.
4. Record why the 100/101-column jump stays. Update `Docs/User_Guide/settings.md`.

## Implementation Notes

At 70 columns and below, Settings shows one pane at a time: the open category with **‹ Categories** above it, or the category rail at full width.
- Measured on dev (bundled CSS, height 30): the compact layout's detail content is the terminal width minus 41 (59 at 100, 30 at 71, 29 at 70, 19 at 60, 0 at 41-40). So `SETTINGS_SINGLE_PANE_MAX_WIDTH = 70` is the widest width that would give the open category under 30 cells. Library's precedent switches below 64; Settings' own numbers set its threshold.
- `settings_screen.py`:
  - `_single_pane_now()` and `_sync_single_pane()` set `settings-workbench-single` and `settings-single-list` on `#settings-workbench` and show or hide `#settings-single-pane-back` ("‹ Categories", or "< Categories" in ASCII glyph mode). The classes are also set at compose time, as the compact classes are.
  - `_select_category` shows the open category. The back control shows the rail and focuses the open category's row.
  - With the rail hidden, `_focus_category` focuses the back control. So the category restore after a switch, and Tab from the navigation bar, land on a visible control.
  - When the window narrows into one-pane mode with focus on a rail row, the rail is the pane shown.
- CSS (`features/_settings.tcss`, rebuilt into `screen_agentic_settings.tcss`): one-pane rules under the compact ones.
- AC#4, the jump stays. The detail pane is 59 cells at 100 columns and 34 at 101 (then one more per column, 64 at 133). It is 52 at 134, where the Scope Inspector takes its 36-column floor. The three-pane layout gives the Scope Inspector 21 cells (36 from 134 columns), and the compact layout hides it. So at any breakpoint value the detail pane loses those cells at once: moving the breakpoint only moves the jump. Removing it means the inspector would have to shrink or hide gradually, which is a layout decision this task does not make. Proposed as a follow-up.
- Tests (`Tests/UI/test_settings_category_sweep.py`, bundled-CSS harness, private profile):
  - `test_the_open_category_keeps_30_cells_from_60_down_to_40_columns[overview|providers-models × 24|30]` (AC#1) also checks the back control's label and that the rail is hidden. Its last step narrows with focus on a rail row and expects the rail.
  - `test_every_category_is_reachable_by_keyboard_one_pane_at_a_time[group × 60|40]` (AC#2). Starting from Tab off the navigation bar, it uses only keys: Shift+Tab to ‹ Categories, Enter, Down/Up, Enter. Each node walks one third of the rail, as the sweep does.
  - `test_full_screen_settings_keeps_all_three_panes[211x44|235x52]` (AC#3) checks the rail, detail and inspector widths measured on dev, (32, 126, 43) and (32, 144, 49), and that the back control is hidden.
- Red at base source (run.sh, `-n 4`, load 34-44): AC#1 `(60, 19)` `assert 19 >= 30`; AC#2 `NoMatches` (no back control). AC#3 is green at base by design (it pins the dev layout). Green after: the whole sweep file, 21 passed at load 56-70.
- Lanes: these nodes are in no PR lane. The sweep file's census entries are per node, and the criteria ask for no lane, so the census is not edited. At load 65-70 the keyboard nodes took 75-124 s each. Listed for the coordinator in the report.
- Size ratchet: `settings_screen.py` 35657 -> 35720 under the owner waiver (`Tests/Architecture/test_module_size_ratchet.py`, dated 2026-11-06).
- Covering run (run.sh, `-n 4`, load 50-79): the Settings and app-chrome files that render Settings at 70 columns or fewer or resize it (agents category, footer hints, personal context, chrome fixes, backup entry, footer shortcuts, sweep, narrow layout, compact fields, nav scroll, the configuration hub's resize node, theme picker, workspace folders). 45 failures, excluding this task's new nodes, were rerun at `fu-base-r11`: 36 fail there too, and the other 9 passed 9 of 9 serially here.
- Docs: `Docs/User_Guide/settings.md` layout table, a **‹ Categories** row.

Files: `tldw_chatbook/UI/Screens/settings_screen.py`, `tldw_chatbook/css/features/_settings.tcss`, `tldw_chatbook/css/screen_agentic_settings.tcss`, `Tests/UI/test_settings_category_sweep.py`, `Tests/Architecture/test_module_size_ratchet.py`, `Docs/User_Guide/settings.md`.
