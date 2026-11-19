---
id: TASK-39134
title: "The Settings detail pane narrows as the terminal widens at 100/101 and 133/134 columns, because the Scope Inspector enters all at once"
status: To Do
created_date: 2026-11-07 02:03
labels:
- follow-up
- settings
- ui
- responsive
- needs-owner-decision
priority: low
references:
- tldw_chatbook/UI/Screens/settings_screen.py
- tldw_chatbook/css/features/_settings.tcss
- Tests/UI/test_settings_category_sweep.py
- Docs/User_Guide/settings.md
dependencies:
- TASK-38953
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 11 of the 2026-10-23 burn-down; checked on dev 2684e5453f, 2026-11-07. TASK-38953's notes (round-11 branch fu/r11-settings) record the measurement and leave the layout decision open. Line numbers are dev's.

- TASK-38953 measured the detail pane at:
  - 59 cells at 100 columns, then 34 at 101;
  - 64 at 133, then 52 at 134.
  At 101 that is less than the compact layout gives at 80 columns.
- The cause is the Scope Inspector, which arrives all at once:
  - at 101 with its 21-cell minimum (`SETTINGS_COMPACT_WORKBENCH_MAX_WIDTH = 100`, `settings_screen.py:972`; `#settings-impact-pane`, `_settings.tcss:61-68`);
  - at 134 with its 36-cell floor (`SETTINGS_INSPECTOR_FLOOR_MIN_WIDTH`, `:977`; `.settings-inspector-floor`, `_settings.tcss:77-79`).
- Moving a breakpoint only moves the jump (TASK-38953's notes).
- TASK-33003.7 (Done) accepted the 133/134 step as a recorded cost. Whether a wider terminal may give the user a narrower detail pane is a layout decision no task has made.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The owner decides whether the detail pane may narrow as the terminal widens across these breakpoints, and if not, how the Scope Inspector enters; the decision is recorded
- [ ] #2 A test measures the detail pane's width at every column from 96 to 140 and matches the decision; it is red on dev if the decision changes behaviour
- [ ] #3 At 211x44 and 235x52 the pane widths are unchanged (`test_full_screen_settings_keeps_all_three_panes` green)
- [ ] #4 `Docs/User_Guide/settings.md` describes the decided layout
<!-- AC:END -->
