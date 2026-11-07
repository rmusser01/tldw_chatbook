---
id: TASK-39134
title: "The Settings detail pane narrows as the terminal widens at 100/101 and 133/134 columns, because the Scope Inspector enters all at once"
status: Done
assignee:
- '@claude'
created_date: 2026-11-07 02:03
updated_date: 2026-11-07 07:30
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
- [x] #1 The owner decides whether the detail pane may narrow as the terminal widens across these breakpoints, and if not, how the Scope Inspector enters; the decision is recorded
- [x] #2 A test measures the detail pane's width at every column from 96 to 140 and matches the decision; it is red on dev if the decision changes behaviour
- [x] #3 At 211x44 and 235x52 the pane widths are unchanged (`test_full_screen_settings_keeps_all_three_panes` green)
- [x] #4 `Docs/User_Guide/settings.md` describes the decided layout
<!-- AC:END -->

## Implementation Plan

1. Measure the detail pane's region width at every column from 96 to 145 on base (a sweep test, red first).
2. Apply the owner's rule with one held width: past 100 columns the detail pane holds the width it had there (rounded up to a token), the layout stays compact until the Scope Inspector fits beside it at its 21-cell minimum, and the inspector then takes each new column until its 36-cell floor, where the hold ends.
3. Keep 211x44 and 235x52 unchanged; rewrite on purpose any pin the decision contradicts; update `Docs/User_Guide/settings.md`.

## Implementation Notes

Owner decision (2026-11-06): "Inspector waits for room: show the Scope Inspector only once the detail pane can keep at least the width it had just below the breakpoint, so widening the terminal never narrows the detail pane."

**Measured on base** (region widths, detail and inspector, run.sh, height 30): the detail pane is the terminal width minus 37 in the compact layout (63 at 100), then 38 at 101 beside a 21-cell inspector, growing to 68 at 133, then 56 at 134 beside the 36-cell floor. (The 59/34/64/52 in the description are content widths, 4 cells less.) The three-pane chrome (rail and gutters) is 42.

**Layout** (`settings_screen.py`, `_settings.tcss`, regenerated `screen_agentic_settings.tcss`):
- `SETTINGS_DETAIL_HOLD_WIDTH = 64`: the 63 cells the pane has at 100 columns, rounded up to the existing `$ds-size-64` token (no new token; at 101 the compact pane is exactly 64).
- `.settings-detail-holds` (`#settings-workbench #settings-detail-pane.settings-detail-holds { width: $ds-size-64; }`) applies from 101 columns up to the floor. It is a width, not a max-width: at 127 columns the 6fr share is 63.75 and rounded down to 63 (measured), which narrowed the pane by one cell.
- `SETTINGS_INSPECTOR_ENTRY_MIN_WIDTH = 42 + 64 + 21 = 127`: below it the layout stays compact (inspector hidden; the spare columns, up to 25, stay empty to the right of the pane). From 127 the inspector shows at 21 cells and its 2fr share takes each new column.
- `SETTINGS_INSPECTOR_FLOOR_MIN_WIDTH = 42 + 64 + 36 = 142` (was 134): the floor and the hold hand over at the same column, where the inspector has just reached 36.
- Result: detail 59..63 at 96..100, 64 from 101 to 142, then one more per column; inspector hidden to 126, 21 at 127, 34 at 140, 36 from 142.
- The empty band (101-126) follows from the rule: the inspector cannot show below 21 cells, and anything the pane took there would be given back when it arrives. This is reported for the owner to see, not decided further.

**AC#2.** `Tests/UI/test_settings_category_sweep.py::test_widening_the_terminal_never_narrows_the_detail_pane` sweeps 96-145 (the criterion's 96-140, plus the floor). It asserts no column narrows the pane, the inspector shows only beside a pane at least as wide as at 100 columns, and it is shown at 140 and at its floor at 145. Red with base sources (run.sh, load 48): `narrowed` non-empty, 63 -> 38 at 101 and 68 -> 56 at 134. Green after.

**AC#3.** `test_full_screen_settings_keeps_all_three_panes[211x44|235x52]` green (rail, detail, inspector 32/126/43 and 32/144/49 unchanged).

**Pin rewritten on purpose.** `Tests/UI/test_destination_visual_parity_correction.py::test_settings_inspector_floor_yields_to_the_detail_pane_when_narrow` (TASK-33003.7) asserted a 36-column inspector at 140 columns; the decision puts the floor at 142, and the inspector is 34 there. Its wide end is now 150 columns. It is a plain test (local `RecoveryRequired`), so it was checked under a temporary `@private_profile_test` marker, then reverted: red at head before the rewrite (`(140, Region(... width=34 ...))`), green after.

**Covering runs** (run.sh, `-n 4`, load 50-125): the 12 Settings files with tests at 101-141 columns, `test_settings_category_sweep.py` and `test_settings_providers_models_card_fits_the_pane.py`, and five named nodes: head 255 failed / 310 passed, `fu-base-r12` 260 failed / 304 passed, no head-only failure (the reds are plain-UI `RecoveryRequired` and pre-existing). Because that hides the plain tests, the nine Settings tests of `test_destination_visual_parity_correction.py` ran under a temporary private marker with head and with base sources: the floor pin above was the only head-only red; 5 fail on both (`#acp-mode-strip`/`#settings-category-strip` too tall, the 140x42 boundary note, `KeyError: 'mcp'`/`'skills'`).

**Docs.** `Docs/User_Guide/settings.md`: the Scope Inspector row and the one-pane row describe the decided layout.

Files: `tldw_chatbook/UI/Screens/settings_screen.py`, `tldw_chatbook/css/features/_settings.tcss`, `tldw_chatbook/css/screen_agentic_settings.tcss`, `Tests/UI/test_settings_category_sweep.py`, `Tests/UI/test_destination_visual_parity_correction.py`, `Tests/Architecture/test_module_size_ratchet.py` (+19, owner waiver), `Docs/User_Guide/settings.md`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`.
