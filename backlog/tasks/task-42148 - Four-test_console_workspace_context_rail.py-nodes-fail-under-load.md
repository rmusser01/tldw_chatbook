---
id: TASK-42148
title: "Four test_console_workspace_context_rail.py nodes fail under load"
status: Done
assignee: ['@claude']
created_date: 2026-11-20 20:06
updated_date: 2026-11-21 20:42
labels:
- follow-up
- tests
- console
- flaky
priority: low
references:
- Tests/UI/test_console_workspace_context_rail.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the library stream (round 18), which saw four nodes fail under load at base with `KeyError: 'workspace-1'` and state-timing assertions. At d85e90bc9e (`-n 4`, through the test wrapper), a whole-file run at load 13-16 failed four nodes: `test_expanded_chats_section_capped_row_marker_surfaces_on_header`, `test_console_f1_tolerates_missing_tree_context_snapshot`, `test_collapsed_chats_section_header_shows_aggregate_unchanged` and `test_combined_menu_change_appearance_opens_picker_with_saved_identity`. A rerun of only those four at load 18-20 failed two: `assert 'Chats' == 'Chats ●'` (line 802) and `assert [] == [{'color': ..., 'icon': ...}]` (line 2440). The `KeyError` did not appear in these runs. TASK-41648 fixed a different `KeyError` in the workspace tree.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The four nodes are run repeatedly under load (at least 10 runs, load recorded), and each failure is attributed to a cause
- [x] #2 Every cause is fixed: by a wait on the product's own ready signal where one exists, otherwise by holding the pushed state; not by a longer pause; and the same repeated run passes (amended by coordinator decision 2026-11-21: a product refresh that restarts itself cannot be ruled out by any wait, so pinning the screen's own state builder is accepted)
- [x] #3 For each changed wait, a negative control that removes the behaviour it pins still fails the test
<!-- AC:END -->

## Implementation Plan

1. Probe the four nodes under load: log every `ConsoleWorkspaceContextTray.sync_state` caller after the test's own injection, and at a failing assertion record the asserted value, whether the tray still holds the injected state, and the settled value 2 s later.
2. Attribute each failure from that evidence (a later production sync overwriting the injected state, or the injected recompose not finished after one pause).
3. Replace the single pause with a bounded wait on the product's own signal for each cause, in one shared helper, so the sibling tests that use the same pattern get it too.
4. Negative control per changed wait; 10 of 10 at `-n 4` under load with the load recorded.

## Implementation Notes

**Cause (AC #1).** Each of these tests waits only for its widget to mount, pushes a hand-built state into the rail, pauses once, and asserts. A production `_sync_console_workspace_context` that lands in between pushes the screen's own state over the test's.
- **Base runs.** The four nodes, 10 runs at `-n 4`, load 44-61 (1-min average): 3 runs failed (f1 three times, capped once). The whole file, 4 runs at load 23-55: siblings that push state the same way failed through the same mechanism (`test_console_f1_renders_markup_looking_selected_label_literally`, `test_zero_result_workspace_search_stays_editable_and_initializes_without_echo`).
- **Probe.** A temporary probe logged the caller of every tray `sync_state`. In the f1 failures the test pushed 7.5-8.3 s into the test, because boot was that slow under load. Startup work (`_sync_native_console_chat_ui` and the `_refresh_workspace_files_availability_snapshot` worker) then pushed the screen's state over it within 0.2 s.
- **Deterministic control.** One production `_sync_console_workspace_context()` right after the push reproduces each reported assertion: capped and collapsed `'Chats' == 'Chats ●'`, f1 `KeyError: 'workspace-1'`, and appearance `[] == [{...}]`. For appearance, the sync must land before the row control is read; the menu then opens for the screen's own draft row, and change_appearance stops at "Send or save this chat". A sync after the menu has opened does not break it.
- **The producer startup does not cover.** The workspace-files availability refresh starts whenever a state build finds its 1 s cache expired. In a stress run (the capped and f1 nodes parametrized 10 times each) with only the startup wait below, 1 of 60 failed: a refresh completed about 20 ms after the wait returned and replaced the capped rows.

**Fix (AC #2).**
- **Startup wait.** The file's `_wait_for_selector` now waits for the mounted selector, then for `_wait_for_console_startup_settled`. That helper waits on the screen's own state: attach reconciled, no Console sync in progress, requested or scheduled, and no worker left in the app's manager, on two checks in a row 0.05 s apart (a finishing worker can start another), with a 15 s bound. It returns as soon as the Console is quiet; it is not a pause. Every test in the file goes through it, and so do the two `test_console_rail_color_grammar.py` tests that import this file's `_wait_for_selector` (both pass under a temporary `@private_profile_test` marker, removed afterwards). The sibling f1 test's ad-hoc `console.workers.wait_for_complete()` is gone, because the helper covers it.
- **Hold (AC #2 as amended 2026-11-21).** No ready signal rules out the availability refresh, which the product restarts on its own after it is ready. So each of the four tests also hands its state to the screen's builder through `_hold_workspace_state`: `console._workspace._build_console_workspace_context_state` returns the pushed state, and a later production sync republishes it instead of replacing it. The tests' assertions and their single pause after the push are unchanged. The startup cause is fixed with a wait. The refresh the product restarts on its own has no ready signal, so it is fixed with the hold, which the coordinator accepted on 2026-11-21 when amending AC #2.

**Evidence.**
- **Same stress run, after the fix:** 80 of 80 passed (4 runs, load 32-60).
- **The four nodes after the fix:** 10 of 10 runs passed at `-n 4`, load 31-57.
- **Whole file, final version, 2 runs at load 37-55:** one passed 51 of 51. The other failed only `test_console_workspace_selector_is_compact_plain_status_row`, which is not one of the four and does not push state. Its own section toggle builds state after the 1 s cache has expired; the refresh recomposes the tray while the test's poll calls `query_one("#console-active-workspace")`, which raises `NoMatches`. The probe caught it about 20 ms after the startup wait, and it failed 1 of 4 base runs too. It is proposed as a follow-up.
- **Whole file after the startup wait alone:** 51 of 51 passed in 4 of 5 runs at load 34-60. The fifth failed f1 (`KeyError`), and that failure is what led to the probe above.

**Negative controls (AC #3).** Each removes the behaviour its test pins and runs against the final test file. All four failed 2 of 2 runs at load 37-46:
- **capped:** the expanded header no longer shows `capped_run_marker` → `'Chats' == 'Chats ●'`.
- **collapsed:** the collapsed header no longer shows `run_marker` → `'Chats' == 'Chats ●'`.
- **f1:** the help panel reads `tray._workspace_tree_context_data` directly, so a missing snapshot raises → `AttributeError`.
- **appearance:** change_appearance returns without opening the picker → `[] == [...]`.

The f1 control is the one the old wait could pass by accident: a late startup sync restores the deleted snapshot.

**Follow-up round (coordinator decision 2026-11-21).**
- **Selector sibling.** `test_console_workspace_selector_is_compact_plain_status_row` polled `query_one("#console-active-workspace")` while its own section toggle's state build could start an availability refresh, and the refresh recomposes the tray. Mid-recompose the row is briefly absent, so `query_one` raised `NoMatches`. The probe caught one landing about 20 ms after the startup wait, and the test had failed 1 of 4 base whole-file runs. It now polls `console.query(...)` and waits for any match with height.
  - 10 of 10 at `-n 4`, load 21-23.
  - Negative control: the status row composed with height 2 fails `assert 2 == 1`, 2 of 2.
- **zero_result sibling.** The first coordinator-requested whole-file run failed `test_zero_result_workspace_search_stays_editable_and_initializes_without_echo` with `'' == 'older'`. That is the same signature the deterministic control (one production sync after the push) gives, so its push now goes through `_hold_workspace_state` too. I did this without being asked, on the strength of that evidence.
  - 10 of 10 at `-n 4`, load 20-27.
  - Negative control: the search input composed without the state's query fails `'' == 'older'`, 2 of 2.
- **Whole rail file, final version, `-n 4`, 2 runs at load 20-25:** one passed 51 of 51.
  - The other failed only `test_conversation_row_renders_custom_icon_in_right_action_slot` (`assert 1 == 2` row controls). That test calls `_set_console_rail_preference(left_open=True)` after the startup wait, and then pushes its rows without the hold: the same post-startup refresh race.
  - It was not in the four nodes or in the coordinator's list. Holding every push site in the file needs a check per test, because some tests depend on a later production sync (for example the star-toggle tests), so it is proposed as a follow-up.
  - Before the zero_result hold, the selector-only version also ran the whole file twice: 51 of 51 in one run, zero_result failing in the other.

**Push-site audit (coordinator decision 2026-11-21, second follow-up).** I read every mounted test in the file and asked one question: does it rely on a later production sync updating the rows it pushed? None of the tests that push state does. The tests that do rely on production syncs never push state, so they have nothing to hold. All 25 push sites, in 24 tests, now go through `_hold_workspace_state`.

| Test | Push | Decision | Why |
|---|---|---|---|
| test_flat_conversations_has_no_cross_owner_starred_aggregate | tray | held | asserts on the pushed rows only |
| test_console_workspace_context_mounts_native_tree_without_legacy_groups | tray | held | asserts on controls the mount renders |
| test_console_f1_exposes_full_selected_tree_label_and_complete_grammar | rail | held | pushed tree, then F1; nothing later |
| test_console_f1_renders_markup_looking_selected_label_literally | rail | held | same shape as above |
| test_console_f1_tolerates_missing_tree_context_snapshot | rail | held (round 1) | see above |
| test_zero_result_workspace_search_stays_editable_and_initializes_without_echo | rail | held (round 2) | asserts on the pushed query |
| test_expanded_chats_section_capped_row_marker_surfaces_on_header | tray | held (round 1) | header from the pushed rows |
| test_expanded_chats_section_visible_row_marker_has_no_header_echo | tray | held | header from the pushed rows |
| test_collapsed_chats_section_header_shows_aggregate_unchanged | tray | held (round 1) | header from the pushed rows |
| test_rail_title_budget_scales_with_terminal_width | tray x2 | held | the budget comes from rail width; the resize's on_resize sync now republishes the same state |
| test_on_resize_alone_regrows_wrap_budget_within_one_pause | tray | held | pins the tray's own on_resize fit pass; the screen's on_resize sync can no longer recompose the tray mid-measure |
| test_sync_state_schedules_exactly_one_deferred_fit_pass | tray | held | its assertions run synchronously after its own push |
| test_console_conversation_star_uses_recognizable_star_glyphs | tray | held | asserts on the pushed rows' glyphs |
| test_console_conversation_star_press_confirms_the_toggle | tray | held | asserts on the toast after the star write, not on rows updating |
| test_console_conversation_star_confirms_an_untitled_conversation | tray | held | asserts on the write and the toast only |
| test_row_actions_stay_reachable_when_local_marks_are_unavailable | tray | held | asserts on the pushed rows' openers |
| test_console_workspace_context_search_controls_keep_stable_ids | tray | held | asserts on the pushed query's controls |
| test_conversation_row_shows_placeholder_when_no_active_conversation | tray | held | asserts on the pushed empty session |
| test_conversation_row_renders_custom_icon_in_right_action_slot | tray | held | asserts on the pushed rows' icons |
| test_conversation_right_action_geometry_stays_contained | tray | held | geometry of the pushed row |
| test_conversation_action_ascii_mode_substitutes | tray | held | labels of the pushed rows |
| test_unpersisted_native_row_keeps_combined_menu_available | tray | held | the pushed native row's control |
| test_combined_menu_change_appearance_opens_picker_with_saved_identity | tray | held (round 1) | the pushed row's menu |
| test_conversation_browser_renders_date_dividers | tray | held | dividers of the pushed rows |
| test_narrow_details_rail_truncates_cleanly_and_keeps_the_full_value | none | left | renders the screen's own details |
| test_console_left_rail_splits_staged_context_from_workspace_context | none | left | asserts on the screen's own defaults |
| test_console_workspace_context_exposes_new_conversation_for_default_workspace | none | left | asserts on the screen's own defaults |
| test_console_workspace_selector_is_compact_plain_status_row | none | left; poll fixed | the screen's own row; it polls `query()` now |
| test_session_tray_shows_workspace_scope_and_new_button | none | left | a registry workspace that the production sync renders |
| test_console_workspace_context_renders_active_workspace | none | left | rendered from the registry by a production sync |
| test_console_workspace_context_renders_server_readiness_handoff_and_acp_contracts | none | left | rendered from the registry and app state by a production sync |
| test_console_workspace_context_renders_markup_titles_literally | none | left | rendered from the registry by a production sync |
| test_console_change_workspace_switches_active_context_and_conversation_rows | none | left | relies on the production sync after a workspace switch |

I had expected the star-toggle tests to need a production sync. They do not: they assert on the toast and the durable write, never on rows changing.

- **Whole file after the audit, `-n 4`:** 51 of 51 in each of 11 runs.
  - 5 runs one after another, load 13-18.
  - 6 runs as two concurrent loops of 3 (8 workers at once), load 16-29.
  - The machine was quieter than in the earlier rounds (which ran at load 30-60), so the concurrent pair was added to raise the load.
- **Custom-icon negative control:** the row action control always gets the placeholder glyph, so the saved appearance is never rendered. The test fails `assert '🧪' in '💬'` (pytest prints the escaped `'#x1F9EA' in '#x1F4AC'`), 2 of 2 at load 16-17.

TASK-38204 (To Do) names the same cause for `test_console_workspace_tray_recompose_guard.py`, which this task does not touch. Its AC #3 asks for one shared settle helper, and other files can import this one.

Files: `Tests/UI/test_console_workspace_context_rail.py`.
