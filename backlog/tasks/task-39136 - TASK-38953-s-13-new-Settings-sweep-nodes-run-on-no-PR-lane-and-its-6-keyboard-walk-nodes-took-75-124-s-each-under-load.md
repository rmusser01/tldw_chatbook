---
id: TASK-39136
title: "TASK-38953's 13 new Settings sweep nodes run on no PR lane, and its 6 keyboard-walk nodes took 75-124 s each under load"
status: Done
assignee: ['@claude']
created_date: 2026-11-07 02:03
updated_date: 2026-11-07 08:10
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
- [x] #1 Recent UI lane shard times are read and recorded, and each new node is measured three times with the "Run the gated Tests/UI slice" step's command
- [x] #2 Every new node has a UI census tail entry that measures about 30 s or less, or the notes record its measured time and why it stays off a lane; never the fast PR contract
- [x] #3 Keyboard reachability of every Settings category at 60 and 40 columns is covered by a node on a PR lane
- [x] #4 The census checker and `Tests/CI` pass
<!-- AC:END -->

## Implementation Plan

1. Read recent UI shard times (read-only `gh`).
2. Measure each of the 13 new nodes three times with the UI step's command.
3. Add each node as a UI census tail entry (about 30 s or less); a node far over 30 s even at low load goes to the new `recovered-suites` job instead.
4. Census checker and `Tests/CI`.

## Implementation Notes

**Recent UI shard times (AC#1).** Slowest finished job per shard over 22 PR runs (42202410121, 2026-11-06 10:24Z, to 42321107301, 2026-11-07 00:55Z; 42209855017 left out, 16 of its shards were red on its own change): 12.38-18.63 min, shards 5 (18.63) and 3 (17.65) the heaviest. TASK-39144 has the per-shard table.

**Each node, 3 runs with the UI step's command** (serial, `--timeout=180 --durations=0`), setup + call + teardown seconds:

| Node | Run 1 (load 50-155) | Run 2 (load 32-51) | Run 3 (load 42-51) |
|---|---|---|---|
| keeps 30 cells [overview-24] | 39.3 | 14.5 | 19.8 |
| keeps 30 cells [overview-30] | 37.7 | 14.1 | 23.5 |
| keeps 30 cells [providers-models-24] | 52.3 | 21.1 | 26.6 |
| keeps 30 cells [providers-models-30] | 58.3 | 24.7 | 24.6 |
| keyboard walk [0-60] | 54.0 | 25.0 | 21.0 |
| keyboard walk [0-40] | 44.1 | 22.9 | 19.3 |
| keyboard walk [1-60] | 46.7 | 31.7 | 21.1 |
| keyboard walk [1-40] | 50.3 | 35.6 | 25.8 |
| keyboard walk [2-60] | 64.3 | 29.4 | 31.2 |
| keyboard walk [2-40] | 55.1 | 30.0 | 29.8 |
| full screen [211x44] | 33.4 | 16.8 | 16.3 |
| full screen [235x52] | 40.1 | 19.0 | 17.3 |
| slash in one-pane mode | 38.9 | 19.7 | 16.4 |

All 13 passed in each run.

**Placement (AC#2, AC#3).** Every node is its own UI census tail entry. At load 32-51 each took 14-36 s, so none is far over 30 s and none went to the recovered-suites job. CI ran this lane's entries at 0.51-0.99 times this machine's loaded serial time (median 0.73, TASK-39144's calibration), so the keyboard walks should take about 15-35 s there. The six keyboard walks cover every category's keyboard reachability at 60 and 40 columns. TASK-39144's re-deal placed the entries by weight (charged at run 1's times, the slowest).

**AC#4.** `scripts/check_ui_pr_gate_census.py`: OK, 1854 entries (floor 1854). `Tests/CI` at `-n 4`: 1228 passed, 1 skipped (2026-11-07, final tree).

**Files.** `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`.
