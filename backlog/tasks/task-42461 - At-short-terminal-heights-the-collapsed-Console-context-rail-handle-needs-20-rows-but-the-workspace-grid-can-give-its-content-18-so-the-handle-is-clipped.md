---
id: TASK-42461
title: "At short terminal heights the collapsed Console context rail handle needs 20 rows but the workspace grid can give its content 18, so the handle is clipped"
status: Done
assignee: ['@claude']
created_date: 2026-11-23 06:38
updated_date: 2026-11-24 01:25
labels:
- follow-up
- console
- css
- layout
priority: medium
references:
- tldw_chatbook/css/features/_console.tcss
- Tests/UI/test_console_shell_regions.py
- Tests/UI/test_console_edge_rail_geometry.py
- tldw_chatbook/Widgets/Library/library_rail.py
- tldw_chatbook/css/tldw_cli_modular.tcss
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 22 of the follow-up burn-down (2026-11-23), by TASK-42416's negative control and its review. Checked on the round-22 stack 336d15338b (dev 2f7790873a plus round 22's stacked branches); paths are under `tldw_chatbook/` unless they start with `Tests/`.

**Why.** Two rules in `css/features/_console.tcss` conflict at the grid's floor:
- `#console-workspace-grid` (lines 38-47) has `min-height: $ds-size-20` (20), a top and a bottom border, and `overflow: hidden`. At its floor its content area is 18 rows.
- `.console-rail-handle` (line 594) and `.console-rail-handle-vertical` (line 635) both have `min-height: $ds-size-20`.

When the context rail is collapsed to its handle and the grid sits at its floor, the handle is 2 rows taller than the grid's content, and the grid clips its bottom 2 rows. That happens when the rows above and below take the rest of a short terminal, for example at 100x24 or 160x24. A user in a short terminal sees the handle with its lower rows cut off. This is the same class of defect as TASK-24605 (an Inspect rail fold indicator that did not render at 80 columns).

**Evidence.** TASK-42416 captured a failing run whose grid was `Region(0, 5, 100, 20)` with 18 content rows beside the 20-row handle; normally the grid is 22 rows with 20 content rows. Its negative control, a temporary edit that squeezed the grid by two rows into painted frames, failed 3 of 4 cases with `console-context-rail-handle escapes workspace content: child=Region(x=0, y=6, width=13, height=20), grid=Region(x=0, y=6, width=100, height=18)`. TASK-42416 fixed its test's startup race, not this stylesheet conflict.

**Shared rule.** `.console-rail-handle-vertical` is also added by the Library rail (`Widgets/Library/library_rail.py:493`), and `css/tldw_cli_modular.tcss` carries the rule app-wide, so a change to it reaches the Library rail handle too.

**Update 2026-11-24 (implementation, measured at dev b31e3e9241).** The grid has no 20-row floor in practice: `chat_screen.py` sets `workspace_grid.styles.min_height = 0` inline (a4c0d4f49d, so an expanded composer can take the grid's rows), which overrides the CSS `min-height: $ds-size-20`. Its content is 14 rows at 100x24 and 160x24, 18 at 100x28 and 19 at 100x29; the 20-row handle fits from 30 rows up. So "the grid's content floor" in AC #2 is 0, and AC #3's sizes include one AC #1 size: the edge-rail live-resize test passes through 100x24 with the handle displayed. AC #3 is amended to say so. Only `.console-rail-handle`'s floor clips: the stacked-label form (`.console-rail-handle-vertical`) is capped by its own `max-height: 100%` and already fits.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A containment test at 100x24 and 160x24, with the context rail collapsed and its handle displayed, asserts that the handle's painted region lies inside the workspace grid's content region; it fails on the current stylesheet
- [x] #2 The handle's minimum height no longer exceeds the grid's content floor, and the containment test passes
- [x] #3 At the sizes the existing shell-region and edge-rail geometry tests use, the handle and grid geometry is unchanged (except at 100x24, an AC #1 size the live-resize test passes through, where the handle now fits the grid): those tests pass unedited. The Library rail handle's geometry is unchanged too, shown by a test or a before/after capture
- [x] #4 The built stylesheets match their sources (the preflight's CSS bundle check passes)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Measure, with mounted Console tests on the production CSS stack (`make_console_pilot`, `TldwCli.CSS_PATH`), the collapsed Context handle's region and the workspace grid's content region at 100x24, 160x24, 100x22..31 and the full-screen 200x50.
2. Add a containment test to `Tests/UI/test_console_edge_rail_geometry.py` (collapse the Context rail, wait for the handle, assert the handle sits inside the grid's content and fills its height, and its open button is the widget at its centre) at 100x24, 160x24 and 100x28..31; show it red on the current stylesheet.
3. Fix it in the stylesheet, scoped to the Console Context handle (`#console-context-rail-handle`) so the Library, Personas and Lab handles that share `.console-rail-handle(-vertical)` keep their geometry; rebuild the bundle.
4. Show the test green; capture the handle and grid geometry before and after at the sizes the shell-region and edge-rail tests use and at 200x50; capture the Library handle before and after; run the covering Console UI files.
5. Update the User Guide's collapsed-rail paragraph; add a census node entry for the new test.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Cause (measured).** Mounted Console probes on the production CSS stack (`make_console_pilot`, `TldwCli.CSS_PATH`), base dev b31e3e9241: with the Context rail collapsed, the handle is `Region(0, 6, 13, 20)` at every height below 30 rows while the grid's content is 14 rows at 100x24 and 160x24, 15/16/18/19 at heights 25/26/28/29, and 20 at 30. The handle's height is `100%` but `.console-rail-handle` floors it at 20 rows, and the grid has no floor of its own: `chat_screen.py` sets the grid's `min_height` to 0 inline (a4c0d4f49d27bdc9b270f41eac0bdf55d530f0cb), overriding the CSS `min-height: $ds-size-20`, so it can give its content anything. The task's "18" was one point on that curve. The stacked-label handle was never clipped: `.console-rail-handle-vertical` also caps it at `max-height: 100%`.

**Fix.** One Console-scoped rule in `css/features/_console.tcss`: `#console-context-rail-handle { min-height: $ds-size-0; }`, the floor of the Context rail it stands in for (the same move TASK-24700 made for the Inspector handle). The handle keeps its 100% height, so it fills the grid at every height and nothing moves at 30 rows and above. The id keeps the shared `.console-rail-handle(-vertical)` rules, and so the Library, Personas and Lab handles, as they were. Bundle rebuilt.

**Test.** `Tests/UI/test_console_edge_rail_geometry.py::test_collapsed_context_handle_fits_the_workspace_grid` at 100x24, 160x24, 100x28, 100x29, 100x30 and 100x31: collapse the rail by click, wait (15 s deadline) for the handle, then assert it lies inside the grid's content region, fills its height, contains its open button, and the button is the widget at its centre. Red on the base stylesheet at 100x24, 160x24, 100x28 and 100x29 (`console-context-rail-handle escapes workspace content: child=Region(x=0, y=6, width=13, height=20), grid=Region(x=0, y=6, width=100, height=14)`), green at 100x30 and 100x31; all six green after.

**Unchanged geometry.** Before/after capture (grid, handle, open button, badge, rails, main column, control bar, composer and every shell child, collapsed and default) at 80x24, 100x30 and 200x50 (full screen): identical apart from the handle's computed min-height. The handle changes only below 30 rows, where it now equals the grid's content (14 rows at 100x24/160x24). Library handle (`LibraryNavigationRailHandle` on `APP_STYLESHEETS`, `ConsolidatedCSSApp`) at 30x12, 30x24 and 30x40: identical before and after (min-height 20, max-height 100%, height = screen). The shell-region and edge-rail files pass unedited.

**Covering runs.** Shell-regions, edge-rail, left-rail, narrow-layout, resize-reflow, partial-resize, compact-access, visual-snapshots, rail-label settings, edge-focus restyle, rail refresh scope and Library style-transition files: 155 passed, 70 failed, every failure on the local `RecoveryRequired` trap. Rerun on the bootstrap profile (a throwaway plugin marking every item): 63 passed, 7 failed. Six of those (visual snapshots) fail the same way on the base stylesheet. The seventh, `test_console_narrow_layout.py::test_console_recovery_controls_resize_bar_one_two_one`, failed 2 of 6 with the fix and 1 of 6 on the base stylesheet (load 72-76): it asserts the control bar's height after one bare pause, at 90x30 where no Context handle shows.

**Seen, not fixed.** The horizontal Context handle's badge (`session`/`workspace`) is never painted, at any size: the button takes `100%` of the handle, so the badge lands on the row below it (at 200x50 the badge is at y=44, the handle ends at y=43, and that cell hits the grid's border). Proposed as a follow-up.

**Files.** `tldw_chatbook/css/features/_console.tcss`, `tldw_chatbook/css/tldw_cli_modular.tcss` (rebuilt), `Tests/UI/test_console_edge_rail_geometry.py`, `Docs/User_Guide/console/chat-basics.md`, `scripts/ui_pr_gate_census.txt` and `scripts/check_ui_pr_gate_census.py`.

**Lane.** The test runs about 64 s serially as one function, over the 30 s per-entry budget, so it is six node entries at the census tail, one per size. Serial seconds with the lane's flags (`-p no:xdist --timeout=180 --durations=0`, through the wrapper, load 62-98 on 18 CPUs; setup + call + teardown): `[100x24]` 11.7, `[160x24]` 9.0, `[100x28]` 9.4, `[100x29]` 10.3, `[100x30]` 11.9, `[100x31]` 11.5. `MINIMUM_FILES` is the measured entry count.
<!-- SECTION:NOTES:END -->
