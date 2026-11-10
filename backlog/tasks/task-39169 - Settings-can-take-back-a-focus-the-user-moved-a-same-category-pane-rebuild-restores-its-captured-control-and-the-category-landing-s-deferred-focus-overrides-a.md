---
id: TASK-39169
title: "Settings can take back a focus the user moved: a same-category pane rebuild restores its captured control, and the category landing's deferred focus overrides a press that lands first"
status: Done
assignee: ['@claude']
created_date: 2026-11-08 06:54
updated_date: 2026-11-10 17:04
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

**Not covered:** a DescendantFocus for a pick in the old panes is handled on the screen's pump; if it arrives after the swap's post-rebuild check, that pick is treated as Textual's reset target and kept rather than restored by id (the old panes are removed early in the rebuild, so the window is narrow).
<!-- SECTION:NOTES:END -->
