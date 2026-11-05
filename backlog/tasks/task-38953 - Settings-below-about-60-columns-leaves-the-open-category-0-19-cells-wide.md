---
id: TASK-38953
title: "Settings below about 60 columns leaves the open category 0-19 cells wide"
status: To Do
created_date: 2026-11-05 21:35
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
- [ ] #1 At every width from 60 down to 40 columns (heights 24 and 30), the open category gets at least 30 content cells, or Settings shows one pane at a time with a labelled way back to the category list; a test measures this and is red on dev (19 cells at 60 columns, 0 at 40)
- [ ] #2 Every category stays reachable by keyboard at those widths, including the way back to the category list
- [ ] #3 At 211x44 and 235x52 the Settings layout is unchanged
- [ ] #4 The detail pane never gets narrower as the terminal gets wider across the 100/101-column breakpoint (on dev: 59 cells at 100 columns, 34 at 101), or the notes record why the jump stays
<!-- AC:END -->
