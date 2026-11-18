---
id: TASK-41772
title: "43 media-series guards in Tests/UI stop at RecoveryRequired during setup, so no PR lane can run them"
status: To Do
created_date: 2026-11-18 16:14
labels:
- follow-up
- architecture-review-2026-10-25
- library
- tests
priority: medium
references:
- Tests/UI/test_library_ingest_analyze_skipped.py
- Tests/UI/test_library_media_trash.py
- Tests/UI/test_library_media_return_settlement.py
- Tests/UI/test_library_media_reader_match_nav_t22209.py
- Tests/UI/test_library_media_reader_no_change_sync_t22208.py
- Tests/UI/test_library_media_reader_scroller_resolution.py
- scripts/ui_pr_gate_census.txt
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Phase 4 media lane PR was prepared on 2026-11-18 on dev 47a05413cb (branch `arch/p4-lanes-media`). It puts on the UI PR gate every test that the TASK-36952.3 slices retargeted or added and that passed 3 of 3. The runs used a CI-like venv and the lane's serial flags. 43 of those guards could not join: each fails in `_build_test_app()` -> `load_settings()` with `RecoveryRequired: raw_source_selection_changed`, in 3 of 3 discovery runs.

They build the app under the per-test profile without the `bootstrap_profile` or `@private_profile_test` marker that every app-building census file carries. The lane PR therefore treats them as red in CI too. `lessons-testing-evidence.md` (TASK-34982) records a gated UI test failing in UI Fast Lane this way. No CI run has confirmed it for these tests: no lane runs them, and nightly last ran on 2026-09-27. Locally, the series could check them only through a temporary private-profile marker probe.

The 43, by file (slice that retargeted them in brackets):
- **`test_library_ingest_analyze_skipped.py` (3.5), all 13:**
  - `test_a_structural_change_during_a_no_fallback_repaint_still_repaints`
  - `test_analyze_outcome_reports_a_raised_exceptions_own_message`
  - `test_analyze_skipped_action_absent_without_a_ready_provider`
  - `test_analyze_skipped_action_appears_with_skipped_items_and_ready_provider`
  - `test_analyze_skipped_run_paints_per_item_outcomes_on_their_own_rows`
  - `test_auto_skipped_ids_are_resolved_not_left_actionable_forever`
  - `test_clear_finished_prunes_the_stale_outcome_for_a_reused_media_id`
  - `test_import_run_that_dies_with_the_screen_names_the_import_run`
  - `test_press_over_a_mixed_set_auto_skips_already_analysed_and_notifies`
  - `test_press_over_an_entirely_already_analysed_set_notifies_and_runs_nothing`
  - `test_pressing_analyze_skipped_does_not_toggle_media_select_mode`
  - `test_pressing_analyze_skipped_runs_the_worker_over_exactly_those_ids`
  - `test_second_analyze_skipped_press_while_running_gets_the_existing_notice`
- **`test_library_media_trash.py` (3.3, 3.4), 10:**
  - `test_media_trash_back_generation_fences_late_reactivated_completion`
  - `test_media_trash_compact_status_folds_after_two_readable_rows`
  - `test_media_trash_disabled_browse_controls_name_mutation_interlock`
  - `test_media_trash_entry_requests_one_independent_initial_page`
  - `test_media_trash_focus_recovers_after_suppressed_canvas_sync`
  - `test_media_trash_footer_advertises_the_restore_and_delete_keys`
  - `test_media_trash_geometry_four_sizes_paints_all_fixed_controls`
  - `test_media_trash_render_retry_matches_recoverable_browse_authority`
  - `test_media_trash_unmount_fences_inflight_restore_completion`
  - `test_trash_status_fold_recaps_the_list_so_actions_stay_inside`
- **`test_library_media_return_settlement.py` (3.1, 3.3, 3.7), 10:**
  - `test_authoritative_content_revision_clamps_once_and_labels_outcome`
  - `test_deadline_uses_one_current_geometry_fallback_and_never_requeues`
  - `test_existing_request_rejects_live_signature_drift`
  - `test_first_frame_media_stage_projects_adaptive_compact_without_legacy_class`
  - `test_live_user_row_focus_fences_earlier_queued_geometry`
  - `test_old_media_deadline_cannot_cancel_new_prompts_list_focus_arm`
  - `test_real_compact_transition_floors_prechange_owner_geometry`
  - `test_same_size_recompose_projects_media_stage_without_reconciliation`
  - `test_trash_return_capture_is_frozen_control_receipt_from_normal_media`
  - `test_viewer_return_capture_is_frozen_receipt_from_normal_media`
- **`test_library_media_reader_match_nav_t22209.py` (3.6), 2:**
  - `test_match_navigation_takes_no_document_pass_per_click`
  - `test_multi_megabyte_document_match_navigation_timings`
- **`test_library_media_reader_no_change_sync_t22208.py` (3.6), 2:**
  - `test_image_item_traversal_wall_time_probe`
  - `test_no_change_traversal_builds_no_preview_and_copies_no_content`
- **`test_library_media_reader_scroller_resolution.py`, all 6** (3.6 retargeted the first three):
  - `test_capture_progress_resolves_the_real_scroller_and_snapshots_its_offset`
  - `test_mode_change_restore_lands_synchronously_on_the_laid_out_body`
  - `test_restore_progress_resolves_the_real_scroller_and_applies_the_saved_offset`
  - `test_match_scroll_moves_the_visible_scroller_after_a_mode_round_trip`
  - `test_match_scroll_resolves_the_real_scroller_and_maps_through_the_wrap_index`
  - `test_read_analysis_read_round_trip_restores_the_rendered_scroll_offset`

The same files hold more test functions that stop the same way and that no slice touched: 24 in Trash, 25 in return-settlement, 5 in t22209 and 1 in t22208.

`test_trash_status_fold_recaps_the_list_so_actions_stay_inside` is also listed in TASK-34000.100 (only in open PR #3423), which records it failing on its own assertion ("Long status never folded.") once it runs. The lane PR's review found about 32 more slice-retargeted tests in 11 Tests/UI files that run on no lane; any of them that stop at RecoveryRequired belong in this list.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each of the 43 tests gets the profile setup it needs (a marker or a fixture) and no longer stops at RecoveryRequired during setup
- [ ] #2 Each of them that then passes 3 of 3 in a serial run with the UI lane's flags is in scripts/ui_pr_gate_census.txt, with the census floor raised to match and the shard budget checked
- [ ] #3 Any of the 43 that still fails after the setup fix has its failure recorded, with its own task or fix named
<!-- AC:END -->
