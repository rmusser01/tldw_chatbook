---
id: TASK-39169
title: "Settings can take back a focus the user moved: a same-category pane rebuild restores its captured control, and the category landing's deferred focus overrides a press that lands first"
status: Done
assignee: ['@claude']
created_date: 2026-11-08 06:54
updated_date: 2026-11-10 21:12
labels:
- follow-up
- settings
- ui
- focus
priority: low
references:
- tldw_chatbook/UI/Screens/settings_screen.py
- Tests/UI/test_settings_panel_scoped_updates.py
dependencies:
- TASK-39046
- TASK-39135
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 12 of the 2026-10-23 burn-down (TASK-39135 and its reviews, round-12 branch fu/r12-settings). Read on the round-12 stack 9091820bf2, 2026-11-08. Round 12 changes this code, so it is named by symbol, not line number.

**Same-category rebuild.**
- The category swap captures `focus_id = self._focused_id_in_category_panes()` at swap start and rebuilds both panes. By review r12s-rev2's measure the rebuild takes 81-1,348 ms, and input keeps running meanwhile.
- When no category landing is pending, the swap then calls `_restore_category_pane_focus(focus_id)`.
- A same-category rebuild has no pending landing. Examples are a Workspaces row click, the show-archived toggle and `_refresh_settings_workspaces_pane`. So `_focus_left_pending_origin()` never withdraws anything.
- A focus the user moved during the rebuild, to the rail filter with "/" for example, is then taken back to the captured control.
- TASK-39135's notes leave this out of scope: "a same-category rebuild is outside this task".

**Deferred landing.**
- The landing calls `_focus_category(category_value)`, which focuses a rail `Button`.
- Textual 8.2.8's `Widget.focus()` does `self.app.call_later(set_focus, self)` (`textual/widget.py:4579-4598`), so the focus lands one callback later.
- A key or click handled between `_focus_category` and that callback is overridden.
- TASK-39046's case, a Review press about 40 ms after a navigation, may partly be this window.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A focus the user moves during a same-category rebuild is kept; a test with a real key and a real click during a held rebuild is red before the change
- [x] #2 A focus the user moves between the landing's `_focus_category` and its deferred `set_focus` is kept; a test is red before the change
- [x] #3 With no input, a same-category rebuild still restores the captured control and a category switch still lands on its rail row; the existing pins stay green
- [x] #4 The tests run on a PR lane (never the fast PR contract), with three timings recorded
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Pin both windows with real input in `Tests/UI/test_settings_panel_scoped_updates.py`: a held same-category rebuild (Workspaces `_refresh_settings_workspaces_pane`) with "/", a click, and no input; and a click delivered right after the landing's `_focus_category` call, before its deferred focus.
2. Same-category rebuild: park focus off the control the rebuild destroys (so Textual's reset, which looks like a user move, never runs); after the rebuild restore the captured control only if nothing has focus, or if the user picked a control in the old panes (restore that one).
3. Deferred landing: focus the rail row (and a restored pane control) with `Screen.set_focus` at once instead of `Widget.focus()`.
4. Keep the existing TASK-39135 pins and the same-category pin green; add the new nodes to the UI lane census; User Guide note; size ratchet row.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Same-category rebuild** (`SettingsScreen._swap_category_panes`). The swap now knows a rebuild is same-category: `_compose_detail_pane_region` stamps `_detail_pane_category`, and a swap with a captured pane focus, no pending landing and an unchanged category parks focus (`set_focus(None)`) before the rebuild. Textual's reset after the focused control's removal then never runs, so a focus found after the rebuild is one the user chose and stands. With nothing focused the captured control is restored, as before. A control the user picks in the old panes during the park (seen in `handle_descendant_focus`) is restored by id instead, since its prune would otherwise reset focus onto the rail's group toggle. A category change is untouched: its stamp differs, so it never parks.

**Deferred landing.** `_focus_category` and `_restore_category_pane_focus` call `Screen.set_focus` at once instead of `Widget.focus()`, which Textual 8.2.8 defers to an app `call_later` (`widget.py:4579-4598`). `Screen.set_focus` already defers its own scroll-into-view, so nothing else moves.

**Tests** (`Tests/UI/test_settings_panel_scoped_updates.py`, private-profile):
- `test_a_focus_the_user_moves_during_a_same_category_rebuild_is_kept[none|key|click]`: the detail pane's rebuild is held, then "/" or a real `pilot.click` on the rail filter. Red on base ae9f9e8d21 for `key` and `click` (`assert 'settings-workspaces-show-archived' == 'settings-category-search'`); `none` green before and after (AC #3).
- `test_a_click_between_the_landing_and_its_deferred_focus_is_kept`: wraps `_focus_category` and forwards MouseDown/MouseUp/Click to the screen right after the original call, exactly as `Pilot.click` does. Red on base (`Button#settings-category-appearance is not SettingsCategorySearchInput`), green after.
- The existing pins stay green: the whole file, 21 passed (`-n 4`, load ~90), including `test_same_category_rebuild_keeps_focus_on_the_control_used`, the TASK-39135 nodes and `test_category_switch_lands_focus_on_the_selected_rail_button`.
- On a PR lane: the four nodes are appended to `scripts/ui_pr_gate_census.txt` (UI lane, not the fast contract), floor raised in `scripts/check_ui_pr_gate_census.py`; no shard replay was run (no CI shard logs locally). Three timings (`-n 4`, the four nodes): 27.6-34.6 s, 27.6-29.1 s and 36.6-39.5 s per node (call), load 87-98.

**Also:** `Docs/User_Guide/settings.md` gains a short paragraph on where focus goes after a pane builds. `Tests/Architecture/test_module_size_ratchet.py`: settings_screen row 35852 -> 35888 (+36, exact; owner waiver).

**Not covered:** a click on Textual's reset target itself, made after the reset while the pane remounts, is read as no move, so the removed control is restored over it. The swap compares focus with the recorded target and cannot see input events. The case is narrow and cosmetic (the user clicked where focus already was); telling it apart would need input-event tracking.
**Review r13p-rev1 fixes (2026-11-10, on 982b101d25).**
- **I1 / M1: telling Textual's reset from a user's move.** The DescendantFocus-based pick branch (`_parked_pane_focus_id` / `_parked_focus_reset_possible`) is gone. `SettingsRegion.recompose` now records the focused control its removal prunes and where Textual's synchronous reset (`App._prune` -> `Screen._reset_focus`) put focus (`focus_reset`). The swap keeps the last such record:
  - a parked rebuild restores the removed control's id when focus is still on the reset target, or on nothing;
  - any other focus is the user's and stands.

  This drops the pick-then-"/" loss, because no reset happens when the pick has already lost focus.
- **TASK-39135 twin, fixed the same way.** `_pending_focus_reset_possible` is cleared unless a reset happened and focus is still on its target. A "/" after a pick in the old panes now withdraws a category landing.
- **M2.** `_restore_category_pane_focus` returns whether it focused a control. When a parked rebuild's control is gone with no replacement, the swap lands on `_focus_category(self.active_category)`. Before, focus ended on nothing.
- **Tests** (`Tests/UI/test_settings_panel_scoped_updates.py`; the 4 nodes are appended to the census tail, floor 1917 -> 1921):
  - `test_a_pick_in_the_old_inspector_during_a_same_category_rebuild[old_pane]` (I1). The inspector's rebuild is held after the detail pane's and the old body is clicked; the rebuilt body takes focus. With the region's record disabled it is red: focus ends on `settings-workspace-row-workspace-default`.
  - `[old_pane_then_key]` (M1). Red on 982b101d25 (`'settings-impact-pane-body' == 'settings-category-search'`).
  - `test_a_same_category_rebuild_that_removes_the_control_lands_on_its_rail_row` (M2). An archived workspace's Archive button is removed. Red on 982b101d25 (`None == 'settings-category-workspaces'`).
  - `test_a_move_after_a_pick_in_the_old_panes_withdraws_the_landing` (TASK-39135 twin). Red on 982b101d25 (landed on `settings-category-appearance`).
  - The whole file now passes, 25 tests. Timings for the four new nodes (`-n 4`, load 35-58): 16.6-17.6 s, 21.9-24.5 s and 22.5-23.8 s per node across three runs.
- **Covering run.** 14 Settings/workspace files plus the wake file: 186 passed, 33 failed. At 982b101d25 the same 33 fail. At base ae9f9e8d21, 31 fail; the other two (`test_folder_actions_keep_feedback_and_recovery_visible[size0-textual-light]` and `test_change_review_unavailable_consent_has_no_toggle`) also fail at base on a rerun, so they are flaky there.
- `Docs/User_Guide/settings.md` now names the rail-row fallback. Size-ratchet row 35888 -> 35905 (+17, exact; owner waiver).

**Review r13p-rev2 fixes (2026-11-10, on a401c9c755).**
- **I1: pinning the moved-since-reset check.** Two probe-based tests, adapted from the reviewer's: the old control is picked during the held rebuild, `Widget.recompose` is held after the removal reset focus (`_hold_mount`), then "/" is pressed. `test_a_move_after_the_reset_in_a_parked_rebuild_stands` and `test_a_move_after_the_reset_in_a_category_switch_withdraws_the_landing[rail_pick]` both pass. With `moved_since_reset` forced False they are red: `'settings-workspaces-show-archived'` and `'settings-category-appearance'`.
- **M1: a switch started from inside the detail pane.** Origins include Overview's "Open Providers & Models", Privacy & Security's jump buttons and Advanced Config's guided-path chips. `_focus_left_pending_origin` now returns `origin is not None` instead of requiring `origin.is_attached`, and its docstring bullet is updated: the swap's reset record already rules out Textual's reset. `test_a_move_while_a_pane_opener_switch_waits_withdraws_the_landing` ("/" while the swap waits on its lock) and `[pane_opener]` were red at a401c9c755 (`'settings-category-providers-models'`) and are green now.
- **M2: removal inside the region's batch.** `SettingsRegion.recompose` now removes inside `async with self.batch()`; the lock is re-entrant, so `super().recompose()` nests. `test_a_settings_region_rebuild_removes_its_children_inside_a_batch` (Overview sync-rows region, batch count at its prune) was red at a401c9c755 (`[0, 1]`) and is green now.
- **M3:** the "Not covered" note now names the remaining limitation (a click on the reset target itself).
- **Results.** Five nodes appended at the census tail; floor 1921 -> 1926. The whole file now passes, 30 tests. Timings (`-n 4`, load 32-46): 14.5-22.7 s per node across three runs.
- **Coverage.** `test_settings_custom_endpoints.py` and `test_settings_advanced_disclosures.py`: 33 passed, 9 failed. The same 9 fail at base ae9f9e8d21 with `RecoveryRequired: raw_source_selection_changed`.
- Size-ratchet row 35905 -> 35910 (+5, exact; owner waiver).
<!-- SECTION:NOTES:END -->
