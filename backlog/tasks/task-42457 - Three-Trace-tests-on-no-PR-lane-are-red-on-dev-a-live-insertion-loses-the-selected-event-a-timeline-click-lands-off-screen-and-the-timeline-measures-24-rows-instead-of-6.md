---
id: TASK-42457
title: "Three Trace tests on no PR lane are red on dev: a live insertion loses the selected event, a timeline click lands off screen, and the timeline measures 24 rows instead of 6"
status: Done
assignee:
- '@claude'
created_date: 2026-11-23 06:38
updated_date: 2026-11-24 00:47
labels:
- follow-up
- trace
- tests
- dev-red
priority: medium
references:
- Tests/UI/test_trace_filters_navigation.py
- Tests/UI/test_trajectory_timeline_integration.py
- tldw_chatbook/UI/Screens/trajectory_screen.py
- scripts/ui_pr_gate_census.txt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 22 of the follow-up burn-down (2026-11-23): TASK-42414's covering runs reported these three nodes red at plain dev 2f7790873a, with dev's Trace screen file and with its own. Re-run on the round-22 stack before its rebase (142e821116): 3 failed in 11.75 s at `-n 3`. Checked on the round-22 stack 336d15338b (dev 2f7790873a plus round 22's stacked branches). Paths are under `tldw_chatbook/` unless they start with `Tests/` or `scripts/`.

**Why.** Neither file is in `scripts/ui_pr_gate_census.txt` or a `derived-artifacts.yml` step, so nothing reports these reds. TASK-42414's scan of 84 failed Derived Artifacts runs (2026-11-20 03:35Z to 2026-11-22 18:09Z) found no Trace or trajectory node failing other than its own, which fits: CI never runs them on a pull request.

**Failing nodes and messages.**
- `Tests/UI/test_trace_filters_navigation.py::test_live_insertion_before_selection_preserves_event_id_filters_and_viewport` (line 548): `AttributeError: 'NoneType' object has no attribute 'event_id'` at line 572. The selected event is gone after a live insertion before it.
- `Tests/UI/test_trajectory_timeline_integration.py::test_click_outside_active_brush_atomically_clears_time_and_selects` (line 318): `textual.pilot.OutOfBounds: Target offset is outside of currently-visible screen region.` from the click at line 343.
- `Tests/UI/test_trajectory_timeline_integration.py::test_timeline_renders_bars_for_timed_snapshot` (line 155): `assert 24 == 6` at line 158, where 24 is `TrajectoryTimeline(id='trajectory-timeline', classes='h-6').size.height` on an 80x24 screen. TASK-42414's implementer suspected the timeline is measured without the bundled stylesheet, so its `h-6` class sets no height. That is not confirmed. A full-height timeline could also push the other test's click target off screen; that link is a guess too.

`test_trace_filters_navigation.py` last changed in a0bdb8ad61 (2026-08-23) and `test_trajectory_timeline_integration.py` in ad8a378ef3 (2026-10-02).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each of the three failures has a recorded cause: test drift after a named product or stylesheet change, or a product defect
- [x] #2 Drift is repaired in the tests with their assertions kept; a product defect is fixed in product code with a test shown red first
- [x] #3 The three nodes pass, alone and in a `-n 4` run of their two files
- [x] #4 The two files run in a PR lane, with their serial time under the lane's flags recorded in the notes
<!-- AC:END -->

## Implementation Plan

1. Reproduce the three reds at base, then trace each to the change that broke it (git history of the tests, the harness and the screen).
2. Repair drift in the tests, keeping every assertion; fix a product defect in the screen with a test shown red first.
3. Run the three nodes alone and their two files at -n 4.
4. Lane both files as whole-file tail entries in the UI census, timing each serially with the lane's flags.

## Implementation Notes

**Cause: test drift in all three; no product defect.**
- `test_timeline_renders_bars_for_timed_snapshot` (`24 == 6`) and `test_click_outside_active_brush_atomically_clears_time_and_selects` (`OutOfBounds`) have one cause. The shared `_mounted` harness in `Tests/UI/test_trajectory_screen.py` was a bare `App`.
  - 82650cc1f7 (TASK-21115, 2026-08-23) moved the timeline's `height: 6` and the filter bar's geometry from `DEFAULT_CSS` to `BUNDLED_CSS`, which only the app bundle loads. It moved ten other harnesses to `ConsolidatedCSSApp`, but not this one.
  - A probe at base: on an 80x24 screen the filter bar was at y=5 and 24 rows tall, and the timeline at y=29 and 24 rows tall, so the click target was off screen. The description's guess was right.
- `test_live_insertion_before_selection_preserves_event_id_filters_and_viewport` (`None.event_id`): 0dd6fc3b14 (2026-09-01) put record rows under a `record:` key prefix. It updated two sibling test files but not this one. `_move_cursor_to_key("event:3")` matched no row and did nothing, so the cursor stayed on the turn header.

**Fix (tests only, every assertion kept).**
- The harness is now `ConsolidatedCSSApp` with `CSS_PATH = BUNDLED_STYLESHEET`. Only `test_trajectory_timeline_integration.py` imports `_mounted`, and `test_trajectory_screen.py`'s own tests pass with it.
- The live-insertion test moves to `screen._record_key(...)`. It now also asserts that the cursor is on `event:3` before the insertion, the check that would have caught this drift.

**Red/green.** At base, the three Trace files at `-n 4` gave 3 failed, 91 passed. After the fix, each node passes alone (9.9 s, 8.0 s, 11.1 s) and the two files pass at `-n 4` (57 passed).

**Lane.** Both files are whole-file entries at the tail of `scripts/ui_pr_gate_census.txt`, and `MINIMUM_FILES` is the measured count. Serial time with the lane's flags (`-p no:xdist --timeout=180 --durations=0 --durations-min=0`), measured on a heavily loaded host:
- `test_trace_filters_navigation.py`: 32 tests in 35.2 s, load 42.6.
- `test_trajectory_timeline_integration.py`: 25 tests in 36.1 s, load 43.0.

**Soak.** 32 of 32 runs green (157 tests each, load 49-89; one combined serial run of the four Trace census files with the lane's flags: 124 passed in 222.9 s at load ~74-85): five Trace files at `-n 4` (`test_trace_responsive`, `test_trace_filters_navigation`, `test_trajectory_timeline_integration`, `test_trajectory_live`, `test_trajectory_screen`).

**Files.** `Tests/UI/test_trajectory_screen.py`, `Tests/UI/test_trace_filters_navigation.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`.

**Lane entries split (coordinator review).**
- UI census entries work to about 30 s each, and these files took 35-48 s serially, so both whole-file entries are now one node id per test function. A parametrized function is one entry that runs all its cases.
- Times are serial with the lane's flags (`-p no:xdist --timeout=180 --durations=0 --durations-min=0`), summed over setup, call and teardown and over every case. They were measured on a heavily loaded host: `test_trace_filters_navigation.py` 47.5 s in all at load 62, `test_trajectory_timeline_integration.py` 44.6 s at load 66.
- Each file's first entry also carries the app's first-test warm-up.
  - `Tests/UI/test_trace_filters_navigation.py::test_trace_filter_state_matches_every_dimension_with_and_semantics`: 3.11 s
  - `Tests/UI/test_trace_filters_navigation.py::test_trace_filter_state_derives_options_and_active_summary`: 0.30 s
  - `Tests/UI/test_trace_filters_navigation.py::test_filter_bar_uses_one_state_owner_in_compact_and_wide_layouts`: 3.85 s
  - `Tests/UI/test_trace_filters_navigation.py::test_wide_filter_bar_paints_visible_and_total_counts`: 1.65 s
  - `Tests/UI/test_trace_filters_navigation.py::test_wide_filter_controls_are_contained_and_every_tab_target_is_visible`: 2.28 s
  - `Tests/UI/test_trace_filters_navigation.py::test_wide_tab_chain_has_no_inert_filter_bar_stop`: 1.14 s
  - `Tests/UI/test_trace_filters_navigation.py::test_focused_timeline_outline_does_not_clip_semantic_content`: 4.23 s
  - `Tests/UI/test_trace_filters_navigation.py::test_compact_filter_action_opens_keyboard_dismissible_dialog`: 1.24 s
  - `Tests/UI/test_trace_filters_navigation.py::test_compact_filter_bar_is_actionable_with_enter_and_paints_cue`: 1.39 s
  - `Tests/UI/test_trace_filters_navigation.py::test_compact_filter_dialog_contains_all_controls_and_actions_at_60x18`: 1.33 s
  - `Tests/UI/test_trace_filters_navigation.py::test_structured_dimensions_filter_the_ledger_and_report_counts`: 1.08 s
  - `Tests/UI/test_trace_filters_navigation.py::test_search_filter_and_timeline_time_range_compose_with_and_semantics`: 1.06 s
  - `Tests/UI/test_trace_filters_navigation.py::test_clear_all_clears_search_structured_filters_and_brush_truth`: 1.09 s
  - `Tests/UI/test_trace_filters_navigation.py::test_navigation_bindings_cover_both_directions_for_every_family`: 0.92 s
  - `Tests/UI/test_trace_filters_navigation.py::test_compact_contextual_hints_fit_without_duplicate_navigation_copy`: 1.09 s
  - `Tests/UI/test_trace_filters_navigation.py::test_next_previous_match_use_filtered_order_and_wrap_explicitly`: 1.15 s
  - `Tests/UI/test_trace_filters_navigation.py::test_error_navigation_preserves_order_and_wraps_across_multiple_members`: 0.98 s
  - `Tests/UI/test_trace_filters_navigation.py::test_anomaly_navigation_expands_collapsed_owning_turn_before_cursor_move`: 1.29 s
  - `Tests/UI/test_trace_filters_navigation.py::test_child_navigation_uses_agent_run_lineage_and_excludes_primary_steps`: 1.36 s
  - `Tests/UI/test_trace_filters_navigation.py::test_anomaly_navigation_wraps_both_directions_on_filtered_records`: 4.64 s
  - `Tests/UI/test_trace_filters_navigation.py::test_live_insertion_before_selection_preserves_event_id_filters_and_viewport`: 1.49 s
  - `Tests/UI/test_trace_filters_navigation.py::test_live_refresh_preserves_disappearing_filter_options_and_select_values`: 1.37 s
  - `Tests/UI/test_trace_filters_navigation.py::test_filter_counts_distinguish_mounted_matches_from_total_matches`: 3.36 s
  - `Tests/UI/test_trajectory_timeline_integration.py::test_timeline_mounts_between_search_and_table`: 6.22 s
  - `Tests/UI/test_trajectory_timeline_integration.py::test_timeline_renders_bars_for_timed_snapshot`: 1.13 s
  - `Tests/UI/test_trajectory_timeline_integration.py::test_timeline_focus_exposes_truthful_keyboard_equivalents_for_mouse_actions`: 0.96 s
  - `Tests/UI/test_trajectory_timeline_integration.py::test_timeline_shows_placeholder_without_timing`: 1.07 s
  - `Tests/UI/test_trajectory_timeline_integration.py::test_escape_clears_active_range_before_dismiss_even_from_ledger`: 1.53 s
  - `Tests/UI/test_trajectory_timeline_integration.py::test_filter_owner_time_clear_also_clears_the_visual_brush`: 1.19 s
  - `Tests/UI/test_trajectory_timeline_integration.py::test_brush_filters_ledger_to_active_records`: 1.31 s
  - `Tests/UI/test_trajectory_timeline_integration.py::test_brush_composes_with_search_as_and`: 1.18 s
  - `Tests/UI/test_trajectory_timeline_integration.py::test_brush_clear_restores_all_rows`: 1.25 s
  - `Tests/UI/test_trajectory_timeline_integration.py::test_widget_drag_seam_posts_brush_that_filters`: 1.40 s
  - `Tests/UI/test_trajectory_timeline_integration.py::test_brush_reveals_collapsed_turn_with_active_children`: 1.73 s
  - `Tests/UI/test_trajectory_timeline_integration.py::test_bar_select_moves_ledger_cursor_and_highlights`: 1.46 s
  - `Tests/UI/test_trajectory_timeline_integration.py::test_click_outside_active_brush_atomically_clears_time_and_selects`: 1.65 s
  - `Tests/UI/test_trajectory_timeline_integration.py::test_keyboard_enter_outside_active_brush_uses_same_selection_transaction`: 1.48 s
  - `Tests/UI/test_trajectory_timeline_integration.py::test_bar_message_hidden_by_search_restores_shared_ledger_selection`: 1.41 s
  - `Tests/UI/test_trajectory_timeline_integration.py::test_timeline_enter_hidden_by_provider_restores_shared_selection`: 1.69 s
  - `Tests/UI/test_trajectory_timeline_integration.py::test_bar_select_pages_in_older_record`: 2.37 s
  - `Tests/UI/test_trajectory_timeline_integration.py::test_ledger_cursor_move_highlights_timeline_bar`: 1.45 s
  - `Tests/UI/test_trajectory_timeline_integration.py::test_live_refresh_feeds_timeline_and_preserves_brush`: 1.43 s
  - `Tests/UI/test_trajectory_timeline_integration.py::test_live_refresh_clears_brush_outside_new_domain`: 1.10 s
  - `Tests/UI/test_trajectory_timeline_integration.py::test_widget_clock_formats_local_time_like_ledger`: 0.64 s
  - `Tests/UI/test_trajectory_timeline_integration.py::test_one_drag_gesture_rebuilds_the_ledger_once`: 1.71 s
  - `Tests/UI/test_trajectory_timeline_integration.py::test_a_drag_that_ends_where_it_paused_still_reaches_the_ledger`: 1.44 s
  - `Tests/UI/test_trajectory_timeline_integration.py::test_a_drag_abandoned_without_a_mouse_up_still_reaches_the_ledger`: 1.50 s
  - `Tests/UI/test_trajectory_timeline_integration.py::test_a_non_drag_brush_still_reaches_the_ledger_immediately`: 1.17 s
