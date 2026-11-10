---
id: TASK-39135
title: "A Settings navigation's pending category focus can take focus from a control the user focused before the category pane swap lands"
status: Done
assignee:
- '@claude'
created_date: 2026-11-07 02:03
updated_date: 2026-11-07 07:39
labels:
- follow-up
- settings
- ui
- focus
priority: low
references:
- tldw_chatbook/UI/Screens/settings_screen.py
- Tests/UI/test_settings_configuration_hub.py
dependencies:
- TASK-38953
- TASK-39046
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 11 of the 2026-10-23 burn-down; checked on dev 2684e5453f, 2026-11-07. TASK-39046 fixed only its test and lists this product edge as a follow-up. Line numbers are dev's; round-11 branch fu/r11-settings leaves this path unchanged.

- A real category switch records a pending rail focus (`_select_category`, `settings_screen.py:25800`).
- The asynchronous pane swap then applies it after its recompose (`:4774-4777`). It prefers it over the `focus_id` captured before the swap (`:4762`, `:4778`), and nothing checks whether the user moved focus in between.
- TASK-39046's probe saw the swap take focus about 40 ms after a Review press, and the pressed control never got it back. The window is the swap's duration, so it grows with load.
- TASK-2831 (To Do) is the opposite race, a pending focus lost too early. Any change here must not bring back that loss.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 When the user moves focus to a control after a navigation but before the pane swap lands (held deterministically in a test), that control keeps focus after the swap; the test is red on dev
- [x] #2 When the user has not moved focus, the navigation's category focus still lands; the existing category switch, deep-link and Theme-list focus pins stay green, named in the notes
- [x] #3 The behaviour rides a deterministic state check, not a timer
<!-- AC:END -->

## Implementation Plan

1. Test first (`Tests/UI/test_settings_panel_scoped_updates.py`, beside the task-1338 rail-focus pin): hold `_category_swap_lock`, press Enter on a rail row (a real switch, so its pending category focus is recorded), then move focus to the rail filter with a real key ("/") or a real click, release the lock, and assert the filter keeps focus once the swap's own focus would have landed (an `app.call_later` queued after the swap). A no-input case asserts the new rail row still gets focus. Show red on base.
2. Fix: a `_withdraw_pending_category_focus()` that clears `_pending_category_focus_value` (and Theme's `_theme_list_focus_pending`), called from the real-input sites, as Library's stage-pick landing is withdrawn: the top of `on_key` (before any key that itself navigates re-arms it; F6 excluded, since it deliberately queues behind the swap's focus) and a `get_focusable_widget_at` override (a click, before Textual focuses its target). No timer.
3. Run the existing category-switch, deep-link, Theme-list and F6-mid-swap focus pins; lane the new nodes as UI census tail entries.

## Implementation Notes

**Fix** (`tldw_chatbook/UI/Screens/settings_screen.py`). `_withdraw_pending_category_focus()` clears the switch's pending rail focus (`_pending_category_focus_value`) and Theme's list landing (`_theme_list_focus_pending`). It is called from the real-input sites, the same pattern as Library's stage-pick landing (`_withdraw_library_notes_pending_focus_handoffs`):
- the top of `on_key`, before any branch that itself switches category (Enter on a rail row, `/` search submit), so a key that switches re-arms the landing after withdrawing the old one. F6 is excluded: `_cycle_workbench_pane` deliberately queues behind the swap's own focus (TASK-2831 window), so it must not withdraw it. Shift+F6 is a priority binding and never reaches `on_key`;
- a `get_focusable_widget_at` override, which Textual 8.2.8 calls for a MouseDown just before it focuses the clicked widget (the TASK-39050 lesson: a screen's `on_mouse_down` runs after the click's focus). It withdraws only when the click lands on a focusable widget.
No timer and no focus-event heuristics (a `DescendantFocus` would also count Textual's own focus reset). TASK-2831's opposite race is untouched: nothing but real input clears the intent.

**AC#1.** `Tests/UI/test_settings_panel_scoped_updates.py::test_a_focus_the_user_moves_before_the_swap_lands_is_kept[none|key|click]`. It holds `_category_swap_lock`, presses Enter on the Storage row (a real switch), moves focus to the rail filter with a real `/` or a real click, releases the lock, and waits for an `app.call_later` queued after the swap, so the swap's own `Button.focus()` (also an `app.call_later`) has landed. Red on base, run.sh `-n 3`, load 50: `[key]` and `[click]` both `assert 'settings-category-storage' == 'settings-category-search'`; `[none]` passed. Green after: 3 passed.

**AC#2.** With no input the switch still lands focus on the new rail row (`[none]`). Existing pins, run.sh `-n 4` at head (load 50-105): `test_category_switch_lands_focus_on_the_selected_rail_button` and the rest of `test_settings_panel_scoped_updates.py`; `test_workbench_pane_focus.py::test_settings_f6_pressed_mid_category_swap_lands_in_the_new_panes`; `test_settings_category_sweep.py` (`test_every_settings_category_renders_and_restores_focus`, 6 nodes); the Theme-list pins in `test_settings_theme_picker.py` and `test_settings_theme_picker_screen.py` (`test_rail_click_and_open_theme_focus_the_theme_list`); `test_settings_deep_link_first_edit.py` (whole file); and `test_settings_configuration_hub.py::test_provider_navigation_conflict_requires_review_discard_or_return` (TASK-39046's Review test). 220 passed, 9 failed. All 9 fail the same way on `fu-base-r12`: 6 plain `test_workbench_pane_focus.py` nodes with `RecoveryRequired: raw_source_selection_changed`, `test_try_then_clean_category_switch_restores_the_pre_editor_theme` (`NoMatches: SelectOverlay`), `test_highlighted_launch_and_override_status_is_on_screen[size0]` (row width), and `test_provider_navigation_conflict_discard_explicitly_applies_staged_target` (failed at base, then passed at head in a three-node rerun at load 105: a load flake of TASK-39046's kind, not this change).

**AC#3.** The withdrawal is a state change at real input; there is no timer.

**Lane.** The three new nodes are UI census tail entries (see TASK-39133's notes for the census edit and job times).

Not changed: when the swap's pending focus is withdrawn, the swap still restores a focus id it captured inside the rebuilt panes (`_restore_category_pane_focus`). On a category change the old category's ids are not in the new panes, so this does not take focus back in practice; a same-category rebuild is outside this task.

Files: `tldw_chatbook/UI/Screens/settings_screen.py`, `Tests/UI/test_settings_panel_scoped_updates.py`, `Tests/Architecture/test_module_size_ratchet.py` (+21, owner waiver), `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`.

**Review r12s-rev1 (I1, M1).** Withdrawing on any key except F6 lost the landing to keys that leave focus where it is (a PageDown mid-swap, Down/PageDown after a filter submit, the detail pane's "Open …" buttons that the swap destroys). The input-event withdrawal and the `get_focusable_widget_at` override are gone. `_select_category` records what had focus with the intent (`_pending_category_focus_from`), and the swap skips the landing (and Theme's list landing) only if focus has left that widget, checked at swap start, where `focus_id` is captured and before the rebuild can reset focus. Clicks need no hook: a click that moves focus changes it before the swap starts.
- The test now starts the switch programmatically (as a deep link does) with focus on the Storage row, so the landing has to move focus, and adds a `[pagedown]` case (laned as a census tail entry). On 00243178b1 `[pagedown]` was red (`'settings-category-storage' == 'settings-category-appearance'`); on base `[key]`/`[click]` were red (`'settings-category-appearance' == 'settings-category-search'`); all four green after.
- Known limit: a move made while the rebuild itself runs (after swap start) is not seen; checking later would also count Textual's focus reset after the rebuild.

**Review r12s-rev2 (I1', M-a).** The swap-start check covered only the 3-8 ms queue gap; the rebuild and landing take 81-1,348 ms (the review's measure) while input keeps running. One predicate, `_focus_left_pending_origin()`, now runs at swap start and again at landing: it counts a move only when the recorded origin is still attached and focus is on another, non-None widget outside `#settings-detail-pane` / `#settings-impact-pane`. A move into the panes the swap rebuilds, or Textual's reset after they are removed, keeps base's landing. When the landing is withdrawn after the rebuild, the captured `focus_id` restore is skipped too, so the user's focus stands. `_withdraw_pending_category_focus()` clears the value, the recorded origin and Theme's list flag wherever the intent is consumed or withdrawn (swap landing, `recompose()`).
- `test_a_focus_the_user_moves_while_the_panes_rebuild_is_kept` holds the detail pane's `recompose`, presses "/" mid-rebuild: red on 6059958f8c (`'settings-category-appearance' == 'settings-category-search'`), green after.
- `test_a_tab_into_the_panes_the_swap_destroys_keeps_the_landing` (Overview "Open Providers & Models", Enter, Tab mid-swap): red on 6059958f8c (focus on `settings-category-group-domain-defaults`), green after.
- Both laned at the census tail (floor 1825). `settings_screen.py` size 35786 -> 35814 and the pre-importer pass 426,525 -> 426,553 LOC (owner waiver, snapshot refreshed).

**Review r12s-rev3 (I1 third round, Minors 1 and 2).**
- Textual's focus reset after the old panes go was counted as a user move when the origin survived (origin a rail row; focus into the OLD detail body during the rebuild; the reset picked the Domain Defaults toggle). `handle_descendant_focus` now sets `_pending_focus_reset_possible` when focus lands in the old panes while a landing is pending (it starts true when the origin is itself in the panes); once set, no move withdraws the landing.
- The predicate also requires `focused.is_attached` (a deferred `focus()` on a removed widget can leave `Screen.focused` detached).
- After this swap rebuilt the detail pane (`_pending_detail_rebuilt`), an attached widget there can only be new content, so focus on it counts as a move at landing (the TASK-39046 Review -> endpoint shape). Only a programmatic focus or a key can reach it then: the swap runs inside a batch, so the new pane has no layout and a real click lands elsewhere (the review's probe: `mid=nav-home` on base and head).
- Tests: `test_a_focus_reset_out_of_the_old_panes_keeps_the_landing` (red on cbf8a2fc47: `'settings-cat...main-defaults' == 'settings-category-appearance'`) and `test_a_control_chosen_in_the_new_pane_while_the_inspector_rebuilds_is_kept` (red on cbf8a2fc47: focus taken to `settings-category-appearance`), both green after and laned (floor 1827). `settings_screen.py` 35814 -> 35852, pre-importer pass 426,553 -> 426,591 LOC (owner waiver).
- Probe results at head (review probes focus_probe2-5, plus a standalone F6 and a deep-link probe; run.sh, load 15-45): PageDown from Storage lands on Appearance; "/" and a click on the filter keep the filter (also mid-rebuild); filter submit lands on its category (with Down or PageDown too); Overview "Open Providers & Models" lands on its rail row (with PageDown or Tab too); focus into the old detail body (queued or mid-rebuild) and a click on old text land on the rail row; a control focused in the new detail pane keeps focus; F6 mid-swap goes to the new detail pane; Theme lands in its list (with PageDown too), "/" keeps the filter; deep links to Storage, Providers & Models and Theme are the same on head and base.
