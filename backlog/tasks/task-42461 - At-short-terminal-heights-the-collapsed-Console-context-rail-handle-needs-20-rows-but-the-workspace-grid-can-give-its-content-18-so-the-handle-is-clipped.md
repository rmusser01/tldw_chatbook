---
id: TASK-42461
title: "At short terminal heights the collapsed Console context rail handle needs 20 rows but the workspace grid can give its content 18, so the handle is clipped"
status: To Do
assignee: []
created_date: 2026-11-23 06:38
updated_date: 2026-11-23 07:09
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
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A containment test at 100x24 and 160x24, with the context rail collapsed and its handle displayed, asserts that the handle's painted region lies inside the workspace grid's content region; it fails on the current stylesheet
- [ ] #2 The handle's minimum height no longer exceeds the grid's content floor, and the containment test passes
- [ ] #3 At the sizes the existing shell-region and edge-rail geometry tests use, the handle and grid geometry is unchanged: those tests pass unedited. The Library rail handle's geometry is unchanged too, shown by a test or a before/after capture
- [ ] #4 The built stylesheets match their sources (the preflight's CSS bundle check passes)
<!-- AC:END -->
