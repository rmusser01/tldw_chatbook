---
id: TASK-42460
title: "The Trace long-inspector test reads the overflow cue after one pause and sometimes finds it empty, on no PR lane"
status: Done
assignee:
- '@claude'
created_date: 2026-11-23 06:38
updated_date: 2026-11-24 00:47
labels:
- follow-up
- trace
- tests
- flaky
priority: low
references:
- Tests/UI/test_trace_responsive.py
- tldw_chatbook/UI/Screens/trajectory_screen.py
- scripts/ui_pr_gate_census.txt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 22 of the follow-up burn-down (2026-11-23), in TASK-42414's covering runs. Checked on the round-22 stack 336d15338b (dev 2f7790873a plus round 22's stacked branches); paths are under `tldw_chatbook/` unless they start with `Tests/` or `scripts/`.

**The flake.** `Tests/UI/test_trace_responsive.py::test_long_inspector_is_focusable_scrolls_to_painted_bottom_and_full_pane` (line 198) failed once at load 47 with `assert '▼ more — scroll' in ''` at line 225, then passed 4 of 4 serially. In 32 repeats at `-n 4` it failed 1 and 0 times with dev's Trace screen file, and 2 and 0 times with TASK-42414's, so it flakes at dev too. The test presses Enter, waits one `pilot.pause()` (line 216), and reads `#trajectory-inspector-overflow`. The assert just before it, `inspector.max_scroll_y > 0` (line 224), passed, so the overflow was there and only the cue text was missing. The cue is written by `_schedule_inspector_cue` (`UI/Screens/trajectory_screen.py:1675`), which runs a second pass 10 ms later against the inspector's layout. Whether the read lands before that pass, or the cue pass runs before layout is final, is not known.

**CI.** The file is in no PR lane: it is not in `scripts/ui_pr_gate_census.txt` and no `derived-artifacts.yml` step names it, so CI has never run it on a pull request. TASK-42414's scan of 84 failed Derived Artifacts runs (2026-11-20 03:35Z to 2026-11-22 18:09Z) found no Trace node failing other than its own. Low priority for that reason; once the file is laned, this becomes a lane flake.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The race is named with evidence: a temporary delay (reverted) in the cue's pass or the inspector's layout makes the failure at line 225 reproducible
- [x] #2 A product race is fixed in the screen, so the cue is right whenever the inspector overflows; a test-timing race is fixed in the test, which waits for the cue with a load-tolerant deadline. The test keeps its assertions
- [x] #3 The test passes in repeated runs of its file under parallel load (at least 32 runs at `-n 4`, or a soak recorded in the notes)
- [x] #4 The file runs in a PR lane, with its serial time under the lane's flags recorded in the notes
<!-- AC:END -->

## Implementation Plan

1. Name the race with a temporary, reverted delay (the cue's pass or the inspector's layout) that makes :225 fail reproducibly.
2. Product race: reconcile the cue in the screen when the inspector's layout lands. Test-timing race: wait for the cue with a deadline in the test. Keep every assertion.
3. Soak the file 32 times at -n 4 under load.
4. Lane the file as a whole-file tail entry in the UI census, timed serially with the lane's flags.

## Implementation Notes

**Race (AC #1).**
- A `call_after_refresh` pass and a 10 ms timer set the cue. While a layout is pending, `Screen._on_idle` does not run queued callbacks; it resumes the update timer, and `_on_timer_update` lays out and then hands them to `call_next`.
- When the update timer is starved, `Pilot.pause()` ends with its own `screen._on_timer_update()`. The test then reads the overflow (`max_scroll_y > 0`) before the queued cue pass has run.
- Control (temporary, reverted): in `_show_inspector`, the screen's update timer was paused and swapped for a no-op for 0.5 s. At base this failed the test at :225 3 of 3, and `test_overflow_cue_reconciles_into_the_100x30_compositor_frame` at :268 3 of 3.
- Ruled out by controls:
  - Delaying only the timer pass to 0.3 s passed 6 of 6.
  - A 50 ms busy pump after scheduling passed 8 of 8, and 64 natural runs at `-n 4` (load ~17) passed.

**Fix.**
- **Product.** The screen subscribes to its own `screen_layout_refresh_signal` (`immediate=True`) and sets the cue in the layout that shows the overflow. A `Static.update` with unchanged text is skipped, because an update lays out again. The two passes and the four calls that scheduled them are deleted.
- **Tests.**
  - With the fix and the same hold, the 100x30 test passes. The 60x18 test passes :225, then fails at its `end` step (:230): Textual's `end` binding is a deferred, animated `scroll_end`, the same race. Its `end` step now waits for `scroll_y == max_scroll_y` with a 10 s deadline. With that, both pass 3 of 3 under the hold.
  - Sibling: at load ~37, `test_inspector_scroll_survives_same_event_refresh_and_resets_on_change` failed 4 in 32 at base and 4 in 32 with the screen fix: 3 at :991 (deferred `scroll_home`, `8 == 0`) and 1 at :975 (`before` read mid-animation after `end`). It now waits for the settled `end` and for the scroll home: 64 of 64 at load 37-39.
  - Every assertion is kept.

**Soak (AC #3).** 32 of 32 runs green (157 tests each, load 49-89; one combined serial run of the four Trace census files with the lane's flags: 124 passed in 222.9 s at load ~74-85) (see TASK-42457), which includes `test_trace_responsive.py`.

**Lane (AC #4).** `test_trace_responsive.py` is a whole-file entry at the tail of the census: 37 tests in 55.6 s serially with the lane's flags, at load 48.8 (heavily loaded).

**Files.** `tldw_chatbook/UI/Screens/trajectory_screen.py`, `Tests/UI/test_trace_responsive.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`, `backlog/docs/lessons-textual.md`.

**Lane entries split (coordinator review).**
- UI census entries work to about 30 s each, so the whole-file entry for `test_trace_responsive.py` is now 31 node ids, one per test function.
- Times are serial with the lane's flags, summed over setup, call, teardown and every case. The file took 68.9 s in all at load 73, on a heavily loaded host.
  - `Tests/UI/test_trace_responsive.py::test_trace_harness_uses_latest_dev_consolidated_production_css`: 3.00 s
  - `Tests/UI/test_trace_responsive.py::test_trace_ledger_uses_in_bounds_responsive_columns`: 6.74 s
  - `Tests/UI/test_trace_responsive.py::test_responsive_rebuild_preserves_stable_event_selection_and_filters`: 1.74 s
  - `Tests/UI/test_trace_responsive.py::test_long_inspector_is_focusable_scrolls_to_painted_bottom_and_full_pane`: 2.74 s
  - `Tests/UI/test_trace_responsive.py::test_overflow_cue_reconciles_into_the_100x30_compositor_frame`: 1.60 s
  - `Tests/UI/test_trace_responsive.py::test_title_uses_trace_and_moves_conversation_id_into_inspector_metadata`: 1.76 s
  - `Tests/UI/test_trace_responsive.py::test_record_kinds_are_humanized_in_ledger_but_raw_kind_remains_in_detail`: 1.13 s
  - `Tests/UI/test_trace_responsive.py::test_explicit_empty_filtered_imported_incomplete_and_no_timing_states`: 5.79 s
  - `Tests/UI/test_trace_responsive.py::test_loading_state_is_visible_until_real_render_worker_lands`: 1.44 s
  - `Tests/UI/test_trace_responsive.py::test_render_failure_is_payload_safe_and_r_retries_real_worker`: 2.00 s
  - `Tests/UI/test_trace_responsive.py::test_render_retry_is_visible_single_flight_and_recovers_after_failure`: 2.60 s
  - `Tests/UI/test_trace_responsive.py::test_successful_render_retry_clears_failure_when_filter_makes_specs_stale`: 1.28 s
  - `Tests/UI/test_trace_responsive.py::test_live_failure_is_visible_and_retry_uses_snapshot_builder`: 2.93 s
  - `Tests/UI/test_trace_responsive.py::test_stale_live_failure_cannot_replace_newer_revision_success`: 1.33 s
  - `Tests/UI/test_trace_responsive.py::test_open_import_marks_the_pushed_production_screen_read_only_shared`: 2.05 s
  - `Tests/UI/test_trace_responsive.py::test_live_following_and_paused_states_are_visible_and_actionable`: 1.17 s
  - `Tests/UI/test_trace_responsive.py::test_empty_ledgers_do_not_advertise_row_only_actions`: 0.93 s
  - `Tests/UI/test_trace_responsive.py::test_projected_generic_retrieval_payload_is_preserved_as_safe_json`: 1.51 s
  - `Tests/UI/test_trace_responsive.py::test_full_detail_keeps_visible_inspector_focus_and_compact_hints`: 1.19 s
  - `Tests/UI/test_trace_responsive.py::test_x_clears_search_and_timeline_brush_together`: 1.32 s
  - `Tests/UI/test_trace_responsive.py::test_search_focus_keeps_x_as_text_until_escape_enables_recovery`: 1.82 s
  - `Tests/UI/test_trace_responsive.py::test_focus_only_transitions_refresh_filter_recovery_truth`: 1.55 s
  - `Tests/UI/test_trace_responsive.py::test_inspector_scroll_survives_same_event_refresh_and_resets_on_change`: 2.07 s
  - `Tests/UI/test_trace_responsive.py::test_empty_modes_and_compound_state_truth_are_painted_at_60_columns`: 2.22 s
  - `Tests/UI/test_trace_responsive.py::test_compact_state_prioritizes_filter_recovery_before_secondary_facts`: 1.11 s
  - `Tests/UI/test_trace_responsive.py::test_paused_live_append_retains_selected_event_outside_newest_page`: 1.39 s
  - `Tests/UI/test_trace_responsive.py::test_legacy_identity_survives_insertion_when_provider_distinguishes_rows`: 1.37 s
  - `Tests/UI/test_trace_responsive.py::test_legacy_identity_survives_pending_to_completed_mutation`: 0.97 s
  - `Tests/UI/test_trace_responsive.py::test_legacy_collision_keys_survive_distinct_collider_inserted_ahead`: 0.93 s
  - `Tests/UI/test_trace_responsive.py::test_six_digit_record_identity_is_fully_painted_without_horizontal_scroll`: 4.84 s
  - `Tests/UI/test_trace_responsive.py::test_record_label_prefers_projection_label_then_generic_sentence_case`: 1.16 s
