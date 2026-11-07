---
id: TASK-39141
title: "Four more test files fail with RecoveryRequired under their plain command: test_library_shell, test_library_ingest_flow, test_console_inspector_rail_minors and test_persistent_terminal_privacy_boundary"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-07 23:59'
created_date: 2026-11-07 02:03
labels:
- follow-up
- tests
- test-infra
priority: medium
references:
- Tests/UI/test_library_shell.py
- Tests/integration/test_library_ingest_flow.py
- Tests/UI/test_console_inspector_rail_minors.py
- Tests/Architecture/test_persistent_terminal_privacy_boundary.py
- Tests/UI/app_factory.py
- Tests/conftest.py
dependencies:
- TASK-38111
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 11 of the 2026-10-23 burn-down; checked on dev 2684e5453f, 2026-11-07. These files are outside the list TASK-38111 (round-11 branch fu/r11-recovery) fixes. Runs were at load 33-54.

| File | Run on dev | Result |
|---|---|---|
| `Tests/UI/test_library_shell.py` | 848 nodes collected; a 4-node sample | Sample: 4 of 4 `RecoveryRequired`, through `Tests/UI/app_factory.py:130` (`load_settings()`). TASK-38111's junit has 785 red: 779 `RecoveryRequired` and 6 on their own assertions (`test_ingest_cta_uses_one_canonical_label_everywhere`, `test_library_export_repeat_press_keeps_the_first_origin`, two generated-stylesheet nodes, the rail scroll CSS node, the note CSS node). The file already has 5 `bootstrap_profile` and 1 `private_profile_test` tests. |
| `Tests/integration/test_library_ingest_flow.py` | 13 nodes | 2 failed, 9 errors, 2 passed. All 11 reds are `raw_source_selection_changed`, through the `library_screen` fixture's `_build_test_app()`. |
| `Tests/UI/test_console_inspector_rail_minors.py` | 5 nodes | 4 failed, all `raw_source_selection_changed`. It imports focus-carrier helpers, but TASK-38111's module marker does not carry over. |
| `Tests/Architecture/test_persistent_terminal_privacy_boundary.py` | the file | 1 red: `test_app_owned_terminal_session_leaves_live_console_boundaries_free` fails in `_build_test_app` with `RecoveryRequired`. It passes under a `bootstrap_profile` marker. |

- None of the four files runs on a PR lane.
- About 795 nodes check nothing in a local one-file run, so a regression in these files is invisible locally.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each file passes, or fails only on its own assertions, under its plain one-file command on a clean checkout; the before and after node counts are recorded, with the nodes still red listed by their own failures
- [x] #2 Each file's cause is read from its own traceback and named in the notes
- [x] #3 No test binds config to the real profile, and RecoveryRequired admission behaviour is unchanged
<!-- AC:END -->

## Implementation Plan

1. Run each of the four files alone on base 8b762b52fa through run.sh at `-n 4`; read each file's own tracebacks and name the cause.
2. Fix with the suite's existing seams, in test code only: `bootstrap_profile` where the tests read config but select no profile of their own; `select_config_source` (binding every module-level config import on the path) where a fixture selects its own config file. No production, conftest or RecoveryRequired change.
3. Re-run each file alone (before/after node counts; still-red nodes listed by their own failures).
4. Prove isolation as TASK-38111's review did: a throwaway path-string audit probe plugin, a check that the scratch HOME holds only what run.sh creates, and ordering runs (reverse and random order).
5. Record each file's timing for the lanes stream under "Proposed follow-ups"; do not lane the files.

## Implementation Notes

**Approach.** These are TASK-38111's seams, used in test code only. No production file, conftest, `Tests/private_profile.py`, `app_factory` or run.sh change. `RecoveryRequired` admission is unchanged.
- `pytestmark = pytest.mark.bootstrap_profile` on `Tests/UI/test_library_shell.py`, `Tests/integration/test_library_ingest_flow.py` and `Tests/UI/test_console_inspector_rail_minors.py`.
- A node-level `bootstrap_profile` marker on `test_app_owned_terminal_session_leaves_live_console_boundaries_free`. The other nodes in `Tests/Architecture/test_persistent_terminal_privacy_boundary.py` are pure.
- Two Library tests selected their own profile by redirecting `TLDW_CONFIG_PATH` and force-reloading, which the bound source refuses under any marker:
  - `test_library_real_config_creation_admits_fresh_profile_to_starter` is now `@private_profile_test`. It deletes its own child profile's config, so the forced load creates it as before. Control: with the delete removed (reverted), it fails on `assert app.library_new_profile_admission is True`.
  - `test_library_real_existing_config_without_lifecycle_defaults_expanded` uses the selected profile's existing config, which is the case it pins. Control: with a `starter` lifecycle stored in place of the pop (reverted), it fails on `assert <LibraryLifecycle.STARTER> is <LibraryLifecycle.EXPANDED>`. Like the original, this test builds its app without `preserve_profile_admission`, so the factory clears admission whatever the creation flag holds.
- I did not use `select_config_source` for the redirect tests. Their path builds and mounts the whole app, whose modules import config functions at module level (`app.py`, `app_factory`, the Library screen and more), and TASK-38111's lesson is that every such binding must be passed.

**Causes, each read from the file's own tracebacks on base 8b762b52fa** (run.sh, one file per command; `-n 4`, the Architecture file serial).
- `test_library_shell.py`: 785 failed / 63 passed. 779 are `RecoveryRequired: raw_source_selection_changed`:
  - 718 through `_build_test_app` -> `app_factory.build_test_app_config` -> `load_settings()`;
  - 44 through `AppFooterStatus.compose` -> `_build_tamagotchi` -> `get_cli_setting`;
  - 15 through `submit_library_ingest_job` -> `_resolve_ingest_backend` -> `get_cli_setting`;
  - 2 through the two tests' own redirect and `load_settings(force_reload=True)`.

  The other 6 failed on their own assertions, as TASK-38111 recorded.
- `test_library_ingest_flow.py`: 2 failed, 2 passed, 9 errors. All 11 reds are `RecoveryRequired`: 9 at setup in the `library_screen` fixture's `_build_test_app()`, 1 through `_resolve_ingest_backend`, and 1 through the test's `_server_backend` helper inside `_expand_library_ingest_source`.
- `test_console_inspector_rail_minors.py`: 4 failed, 1 passed. All 4 are `RecoveryRequired` through `_build_test_app()`. The focus-carrier module's marker does not carry over to a file that imports its helpers.
- `test_persistent_terminal_privacy_boundary.py`: 1 failed, 7 passed. The red is `RecoveryRequired` through `_build_test_app()`.

**After, on head** (same commands): `test_library_shell.py` 115 failed / 733 passed. That was 88 minutes at load 53-106; at load 33-57 (the forward ordering run) it was 91 / 775 for three files together. `test_library_ingest_flow.py` 3 failed / 10 passed. `test_console_inspector_rail_minors.py` 5 passed. `test_persistent_terminal_privacy_boundary.py` 8 passed.. No node fails with `RecoveryRequired`.

**Nodes still red on their own assertions: 86, all stale.** 83 are in `test_library_shell.py` and 3 in `test_library_ingest_flow.py`. Each one:
- failed in two whole-list reruns on head;
- failed alone in its own pytest process on head;
- also fails on base 8b762b52fa under a scratch plugin that adds the same marker (`R11R_MARK`). 90 of the 98 head-run reds failed there, including all 86; the other 8 are the load-sensitive ones below.

So the profile wall hid them; this branch did not cause them. Two of them (`test_background_recompose_focus_fallback_stays_inside_screen_content`, `test_library_onboarding_positive_wins_while_another_owner_hangs`) passed once, in head2 at load ~100, and were red in every other run. Grouped by their own failure:
- `textual.css.query.NoMatches` (8): `test_library_note_60x20_navigator_state_allocation[filtered-empty-expected1-#library-notes-filter-clear]` ('#library-notes-empty'), `test_library_ordinary_emergency_retains_bar_and_route_host` ('#library-emergency-return'), `test_library_shell_export_choose_destination_pushes_file_save_dialog_with_sanitized_name` ('#label' on SelectCurrent), the three `test_library_shell_note_export_pushes_file_save_dialog[...]` cases ('SelectOverlay' on Select '#listing-sort'), `test_library_starter_production_geometry_and_focus_order[size0]` and `[size1]` ('#library-row-ingest-import-media')
- `AssertionError: ordinary custom width did not use the canvas-preserving bound` (1 more): `test_ordinary_rail_restores_custom_owner_after_collapse_and_adaptive_route[48]`
- `AssertionError: assert 29 == 24` (5): `test_library_production_width_matrix_graduated_startup_landing[100-24]`, `test_library_production_width_matrix_graduated_startup_landing[80-24]`, `test_library_production_width_matrix_ordinary[100-100-24]`, `test_library_production_width_matrix_ordinary[120-116-24]`, `test_library_production_width_matrix_ordinary[80-80-24]`
- `AssertionError: Ordinary route did not reclaim emergency geometry` (4): `test_adaptive_routes_never_receive_ordinary_emergency_geometry[browse-conversations-#library-conversations-reader-shell]`, `test_adaptive_routes_never_receive_ordinary_emergency_geometry[browse-media-.library-media-route]`, `test_adaptive_routes_never_receive_ordinary_emergency_geometry[browse-notes-.library-notes-route]`, `test_adaptive_routes_never_receive_ordinary_emergency_geometry[browse-prompts-#library-prompts-reader-shell]`
- `AssertionError: assert False` (4): `test_library_media_entry_focus_survives_three_chained_recomposes`, `test_library_media_tab_away_survives_a_background_recompose_within_the_settle_window`, `test_library_ordinary_emergency_route_matrix_focus_and_return`, `test_library_starter_hidden_route_focuses_compact_rail_without_search`
- `AssertionError: ordinary custom width did not use the canvas-preserving bound` (3): `test_ordinary_rail_restores_custom_owner_after_collapse_and_adaptive_route[24]`, `test_ordinary_rail_restores_custom_owner_after_collapse_and_adaptive_route[34]`, `test_ordinary_rail_restores_custom_owner_after_collapse_and_adaptive_route[35]`
- `AssertionError: assert False is True` (2): `test_library_emergency_real_guards_keep_specific_action_authoritative[cancel]`, `test_library_note_compact_deep_link_intent_opens_notes_stage[context0-#library-notes-filter-navigator-True]`
- `AssertionError: entering the list must focus its first row` (2): `test_library_media_list_focuses_first_row_and_arrow_keys_move_it`, `test_library_notes_list_focuses_first_row_and_arrow_keys_move_it`
- `AssertionError: ('#library-note-body', Region(x=10, y=14, width=50, height=5))` (2): `test_library_note_60x20_editor_state_allocation[normal-expected0-#library-note-body]`, `test_library_note_60x20_untouched_new_allocation_keeps_discard_visible`
- `AssertionError: ('#library-notes-list', Region(x=5, y=13, width=50, height=6))` (2): `test_library_note_60x20_navigator_state_allocation[normal-expected0-#library-notes-filter]`, `test_library_note_60x20_navigator_state_allocation[sort-choice-expected2-#library-notes-sort-newest]`
- `AssertionError: assert NoteEditorInput(id='library-note-keywords') is NoteEditorTextArea(i` (2): `test_library_note_keyboard_focus_order_and_persistent_labels[terminal_size0]`, `test_library_note_keyboard_focus_order_and_persistent_labels[terminal_size1]`
- `AssertionError: assert '0 of 0' not in '           ...            '` (2): `test_library_paged_empty_recovery_is_painted_and_keyboard_reachable[media-size0]`, `test_library_paged_empty_recovery_is_painted_and_keyboard_reachable[media-size1]`
- `assert 36 == 31` (2): `test_library_production_width_matrix_normalizes_persisted_custom_widths[None-31]`, `test_library_production_width_matrix_normalizes_persisted_custom_widths[wide-31]`
- `AssertionError: Expected 'mock' to have been called once. Called 0 times.` (2): `test_library_shell_ingest_canvas_invalid_path_notifies_and_submits_nothing`, `test_library_shell_open_in_console_triggers_handoff`
- `assert None is not None` (1): `test_inline_consent_gates_start_when_pdf_deps_missing`
- `assert 0 == 1` (1): `test_job_persists_to_db`
- `IndexError: tuple index out of range` (1): `test_unsupported_file_not_retryable`
- `AssertionError: the fallback left focus at None` (1): `test_background_recompose_focus_fallback_stays_inside_screen_content`
- `AssertionError: assert None is LibraryRailRowButton(id='library-row-browse-conversations',` (1): `test_background_recompose_restores_focus_on_an_empty_conversations_list`
- `AssertionError: #library-media-title` (1): `test_generated_stylesheet_includes_library_media_rules`
- `AssertionError: #library-header-line` (1): `test_generated_stylesheet_includes_library_shell_rules`
- `AssertionError: assert 'Library: Chunking Lab' == 'Library: Import…'` (1): `test_ingest_cta_uses_one_canonical_label_everywhere`
- `AssertionError: assert 2 > 2` (1): `test_library_canvas_entry_defeats_an_older_emergency_receipt`
- `AssertionError: assert {'archive_sco...set': 20, ...} == {'limit': 20,...y': None, ...}` (1): `test_library_conversations_next_loads_second_service_page`
- `AttributeError: 'types.SimpleNamespace' object has no attribute '_screen'` (1): `test_library_export_repeat_press_keeps_the_first_origin`
- `AssertionError: assert None == {'row_id': 'browse-collections', 'scope': {}, 'source_list_` (1): `test_library_landing_continue_receipt_accepts_only_authoritative_source_scopes[browse-collections-expected_scope4]`
- `AssertionError: ('#library-note-body', Region(x=10, y=14, width=50, height=2))` (1): `test_library_note_60x20_editor_state_allocation[conflict-expected2-#library-note-conflict-copy]`
- `AssertionError: ('#library-note-context-region', Region(x=10, y=11, width=50, height=8))` (1): `test_library_note_60x20_editor_state_allocation[context-expected5-#library-note-context-keywords]`
- `AssertionError: ('#library-note-title-row', Region(x=0, y=0, width=0, height=0))` (1): `test_library_note_60x20_editor_state_allocation[delete-confirmation-expected3-#library-note-delete-cancel]`
- `AssertionError: ('#library-note-body', Region(x=10, y=16, width=50, height=3))` (1): `test_library_note_60x20_editor_state_allocation[validation-expected1-#library-note-title]`
- `AssertionError: assert None == 'browse-notes'` (1): `test_library_note_compact_stage_drills_in_and_back_without_losing_origin`
- `assert 7 == 9` (1): `test_library_note_compact_surplus_allocation_expands_only_named_owner[editor-#library-note-body-fixed_selectors1-expected_fixed_heights1-9-15]`
- `assert 8 == 11` (1): `test_library_note_compact_surplus_allocation_expands_only_named_owner[navigator-#library-notes-list-fixed_selectors0-expected_fixed_heights0-11-17]`
- `AssertionError: assert '#library-note-body {' in '/* =====================================` (1): `test_library_note_css_bounds_editor_body_and_mutes_meta`
- `AssertionError: assert 7 == 10` (1): `test_library_note_fifty_same_side_resize_sequences_do_zero_notes_work[initial_size0-sequence0-True]`
- `AssertionError: Footer did not settle to 'n new | / find | g folder | esc rail'; got '/ fo` (1): `test_library_note_footer_covers_navigator_create_sync_and_exit`
- `AssertionError: Footer did not settle to 'esc notes | ctrl+end end'; got 'esc notes | typi` (1): `test_library_note_footer_tracks_editor_states_and_ancillary_contents`
- `AssertionError: assert 'browse-notes' == ''` (1): `test_library_note_local_shortcuts_are_region_scoped_and_flush_guarded`
- `AssertionError: assert True is False` (1): `test_library_note_pilot_delete_pending_locks_and_cancel_restores_context`
- `AssertionError: assert 1 == 0` (1): `test_library_note_recompose_and_fifty_route_cycles_return_to_baseline`
- `AssertionError: assert (('i', 'impor... 'next pane')) == (('i', 'impor... 'next pane'))` (1): `test_library_onboarding_back_requires_fresh_empty_evidence_and_focuses_import`
- `AssertionError: assert 6 == 5` (1): `test_library_onboarding_positive_wins_while_another_owner_hangs`
- `AssertionError: assert 36 == 31` (1): `test_library_production_width_matrix_ordinary[170-166-31]`
- `AssertionError: assert 39 == 34` (1): `test_library_production_width_matrix_ordinary[235-231-34]`
- `AssertionError: #library-rail does not open a CSS rule` (1): `test_library_rail_css_scrolls_vertically_with_scrollbar_styling`
- `AttributeError: 'LibraryScreen' object has no attribute '_request_library_notes_tree_refre` (1): `test_library_resize_geometry_high_frequency_does_no_non_layout_work`
- `AssertionError: assert 'save or discard first' == 'back to list'` (1): `test_library_retained_prompt_editor_keeps_back_authoritative`
- `AssertionError: assert 5 <= 1` (1): `test_library_route_matrix_keeps_default_ordinary_rail_edge_stable`
- `AssertionError: assert '' == 'cake'` (1): `test_library_search_typed_text_survives_registry_recompose`
- `AssertionError: A session-blank note that was autosaved with content mid-session, then emp` (1): `test_library_shell_blank_note_autosaved_then_emptied_still_gcs_on_back`
- `assert [4747994672, 4760169520] == [4760169520]` (1): `test_library_shell_media_viewer_inplace_large_document_latency_and_parse_proxy`
- `AssertionError: assert ['Reading lis...ited)', '17w'] == ['Reading lis...ited)', 'now']` (1): `test_library_shell_note_save_then_back_refreshes_list_title_and_age`
- `assert False` (1): `test_library_shell_restored_export_canvas_rekicks_counts_worker_on_mount`
- `AssertionError: assert None is not None` (1): `test_library_shell_restored_media_viewer_fetches_detail_on_mount`
- `AssertionError: #library-media-viewer-error never mounted within 30.0s (912 polls). Visibl` (1): `test_library_shell_restored_media_viewer_with_deleted_item_shows_recovery`
- `AssertionError: loader made no config reads` (1): `test_options_loader_never_calls_get_cli_setting_without_default`
- `AssertionError: semantic short title missing: '  Conversations (2)'` (1): `test_rail_counts_never_clip_and_titles_shrink_first_at_100x30`
- `AssertionError: #library-rail rule missing from _agentic_terminal.tcss` (1): `test_rail_shows_a_visible_scrollbar_when_content_overflows`
- `AssertionError: assert '#777777' == '#2d2d2d'` (1): `test_search_placeholder_fits_and_input_reads_as_a_field_at_100x30`

**Intermittent under load** (load 33-160, up to six agents on 18 CPUs). 44 nodes, not counted with the 86 above. Each was red in at least one of the four whole-file runs (head1 and head2 are one-file runs; fwd and rev are the ordering runs) and passed in the others, alone, or on a rerun. `test_library_shell_blank_note_escape_key_returns_to_list_without_crash` failed 4 of 5 recent head runs (`NoMatches '#library-notes-status'`) and 1 of 2 on base under the marker. Head2 at load ~100 produced 34 of these; 25 passed alone at once, and the other 9 passed in both ordering runs at load 33-57.
- `test_arming_clear_finished_disturbs_nothing_and_dead_zone_holds` (red in: head2)
- `test_background_recompose_restores_focus_on_a_filtered_empty_media_list` (red in: head2)
- `test_browse_row_error_callout_keeps_the_right_tint_across_a_retry` (red in: head2)
- `test_conversation_canvas_scrolls_current_page_while_pager_stays_fixed` (red in: head2)
- `test_library_conversation_non_entry_locator_is_fenced_by_rail_navigation[warm-page-2]` (red in: head1, head2)
- `test_library_conversation_page_failure_keeps_last_successful_rows[size0]` (red in: head2)
- `test_library_conversation_page_failure_keeps_last_successful_rows[size1]` (red in: rev)
- `test_library_conversation_retry_first_failure_has_no_applied_metadata` (red in: head2)
- `test_library_emergency_ingest_consent_syncs_return_truth` (red in: head2)
- `test_library_graduation_late_restore_does_not_steal_newer_user_focus` (red in: head1, head2)
- `test_library_ingest_progress_tick_updates_only_reserved_line_in_place` (red in: head1, rev)
- `test_library_lifecycle_explore_persists_without_changing_sections` (red in: head2)
- `test_library_media_complete_facets_commit_during_initial_recovery[error-ALL]` (red in: head2)
- `test_library_media_durable_mutation_gates_and_refreshes_applied_scope[False]` (red in: head1)
- `test_library_media_edit_save_stays_gated_through_detail_refresh` (red in: head2)
- `test_library_media_stale_page_disables_actions_across_recompose` (red in: head1)
- `test_library_note_breakpoint_round_trips_restore_every_region_focus_role` (red in: head1)
- `test_library_note_database_purpose_round_trips_at_breakpoint_without_recompose` (red in: head2)
- `test_library_note_deliberate_scroll_override_replaces_responsive_memory` (red in: head1, fwd, rev, head2)
- `test_library_note_fifty_cycle_breakpoint_and_presentation_lifecycle` (red in: head2)
- `test_library_note_fifty_same_side_resize_sequences_do_zero_notes_work[initial_size1-sequence1-False]` (red in: head2)
- `test_library_note_load_failure_outranks_rail_focus_on_compact_entry` (red in: head2)
- `test_library_note_pilot_conflict_can_originate_from_every_region[preview]` (red in: fwd)
- `test_library_note_remount_restores_only_persistent_view_state[terminal_size1]` (red in: head2)
- `test_library_note_same_side_resize_does_no_presentation_work[initial_size0-same_side_sizes0-True]` (red in: head2)
- `test_library_note_same_side_resize_does_no_presentation_work[initial_size1-same_side_sizes1-False]` (red in: head2)
- `test_library_note_unsafe_session_outranks_rail_focus_on_compact_entry` (red in: head2)
- `test_library_note_wide_deep_link_back_clears_future_compact_intent` (red in: head2)
- `test_library_onboarding_explore_during_read_still_allows_graduation` (red in: head2)
- `test_library_onboarding_unmount_flushes_latest_lifecycle` (red in: head2)
- `test_library_rag_scope_recovery_steady_state_snapshot_causes_no_churn` (red in: fwd)
- `test_library_shell_blank_note_escape_key_returns_to_list_without_crash` (red in: fwd, rev)
- `test_library_shell_blank_note_untouched_is_gc_from_real_db_on_back` (red in: head2)
- `test_library_shell_blank_note_untouched_is_gc_from_real_db_on_rail_switch` (red in: head1, fwd, head2)
- `test_library_shell_blank_title_save_round_trip_agrees_with_the_row` (red in: head1, head2)
- `test_library_shell_history_manual_expand_survives_unrelated_refresh` (red in: head2)
- `test_library_shell_ingest_canvas_submit_clears_path_and_title_keeps_metadata` (red in: head2)
- `test_library_shell_media_rail_reentry_restores_the_open_item_and_back_lists` (red in: head2)
- `test_library_shell_media_viewer_shows_analysis_from_latest_version` (red in: head2)
- `test_library_shell_note_back_flushes_while_previewing` (red in: rev)
- `test_library_shell_note_copy_calls_clipboard_seam[#library-note-copy]` (red in: head2)
- `test_library_shell_note_explicit_save_calls_seam_and_bumps_version` (red in: rev, head2)
- `test_library_shell_note_flush_on_back_saves_before_view_switches` (red in: head1, head2)
- `test_slash_focuses_the_prompts_filter_not_the_rail_search` (red in: rev)

**AC#3 isolation evidence.**
- A throwaway path-string probe (TASK-38111 review's `r11r_probe`, loaded through `PYTEST_PLUGINS` from a scratch directory, never committed) recorded every test's profile environment, config cache source and raw-participant paths. An audit hook flagged any path event under the real home outside the repo.
- Two full runs of the three non-Architecture files together at `-n 4`: forward order, and reversed with the review's `r11r_reverse` plugin. Each wrote 868 rows across 7 processes, including the 2 private-profile children, which inherit the plugins through the environment.
  - Every row had HOME, `XDG_*` and `TLDW_CONFIG_PATH` under the bootstrap profile.
  - The config source was bootstrap (804) or not yet loaded (64).
  - 0 raw participants were under the real home, and 0 real-home path events were flagged.
- The Architecture file, run serially in both orders with the same probe: 8 rows each, 0 flagged, 0 real-home participants.
- After every run, the scratch HOME held only the five directories run.sh creates.
- Ordering: forward 91 failed / 775 passed; reversed 93 / 773. Both contain all 86 stale nodes. The 5 (forward) and 7 (reversed) extras are the intermittent nodes above; rerun alone twice, each passed at least once.

**Timings, for the lanes stream** (not laned here; see "Proposed follow-ups"). Run.sh one-file runs at `-n 4`, with the Architecture file serial.
- `test_library_shell.py` (848 nodes): 992 s on base (load 45-49, 785 nodes fail fast); 2,073 s on head at load 45-160 and 5,267 s at load ~100. The three files together took 1,302-1,365 s, two runs side by side at load 33-57. That is far over one UI census entry (about 30 s), so the file would have to be split by node across many shards, and its 86 stale nodes re-pinned first.
- `test_library_ingest_flow.py`: 17-27 s, and 63 s at load ~100.
- `test_console_inspector_rail_minors.py`: 12-24 s, and 51 s at load ~100.
- `test_persistent_terminal_privacy_boundary.py` (serial; it would belong in the Architecture guard job): 42-121 s, and 151 s at load ~100.
