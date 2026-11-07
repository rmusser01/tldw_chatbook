---
id: TASK-39136
title: "TASK-38953's 13 new Settings sweep nodes run on no PR lane, and its 6 keyboard-walk nodes took 75-124 s each under load"
status: To Do
created_date: 2026-11-07 02:03
labels:
- follow-up
- settings
- tests
- ci
priority: low
references:
- Tests/UI/test_settings_category_sweep.py
- scripts/ui_pr_gate_census.txt
- scripts/check_ui_pr_gate_census.py
dependencies:
- TASK-38953
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 11 of the 2026-10-23 burn-down; checked on dev 2684e5453f, 2026-11-07, and on round-11 branch fu/r11-settings, which lands TASK-38953.

- TASK-38953 adds 13 nodes to `Tests/UI/test_settings_category_sweep.py`:
  - `test_the_open_category_keeps_30_cells_from_60_down_to_40_columns` (4 nodes);
  - `test_every_category_is_reachable_by_keyboard_one_pane_at_a_time` (6 keyboard walks: 3 groups × widths 60 and 40);
  - `test_full_screen_settings_keeps_all_three_panes` (2 nodes);
  - `test_slash_in_one_pane_mode_opens_the_rail_with_its_filter` (1 node).
- They pin the one-pane layout, keyboard reachability of every category, and the full-screen widths.
- The branch leaves the census and `derived-artifacts.yml` untouched. On dev the census lists only older per-node entries for this file (`scripts/ui_pr_gate_census.txt:2031-2037`, `:2364` on). So no PR lane runs the new nodes.
- TASK-38953's notes: at load 65-70 the keyboard walks took 75-124 s each, against UI census entries kept near 30 s. The other 7 nodes have no recorded times.
- Precedent: TASK-36525 put the sweep's earlier (group, size) nodes on the census at 11-23 s each.
- The coordinator's list said "10" keyboard-walk nodes; the branch has 6.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Recent UI lane shard times are read and recorded, and each new node is measured three times with the "Run the gated Tests/UI slice" step's command
- [ ] #2 Every new node has a UI census tail entry that measures about 30 s or less, or the notes record its measured time and why it stays off a lane; never the fast PR contract
- [ ] #3 Keyboard reachability of every Settings category at 60 and 40 columns is covered by a node on a PR lane
- [ ] #4 The census checker and `Tests/CI` pass
<!-- AC:END -->
