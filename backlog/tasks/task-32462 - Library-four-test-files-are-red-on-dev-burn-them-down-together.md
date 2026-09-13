---
id: TASK-32462
title: 'Library: four test files are red on dev — burn them down together'
status: To Do
assignee: []
created_date: '2026-09-12 00:12'
labels:
  - library
  - tests
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Measured while verifying the task-32302/32393/32306/32303 branch: four Library test files are red on `origin/dev` itself, 36 failures between them. Each was confirmed at the dev tip with the branch's own production files reverted, so none of them belongs to that branch — but together they mean a Library change cannot be judged by "did the suite go red", which is exactly the signal the next wave needs. One owner should burn them down in one pass, because two of the four share a cause shape (a pin written against copy or a guard that has since moved).

**`Tests/UI/test_library_footer_focus.py` (4)** — every failure is `AttributeError: 'types.SimpleNamespace' object has no attribute '_library_narrow_stage_return_active'`: the fake self these tests pass to `LibraryScreen._library_footer_shortcuts_for_current_state` predates the narrow-stage gate task-32225 added to that method.

- test_typing_focus_drops_swallowed_printable_keys
- test_text_area_focus_gets_the_same_honesty
- test_non_typing_focus_keeps_the_full_set
- test_no_focus_keeps_the_full_set

**`Tests/UI/test_post_release_workspaces_library_depth.py` (2 at dev)** — the seeded cross-workspace sources never register, so the Workspace ▸ Handoff row stays on "Handoff · unavailable until sources exist" and the "Use in Console" tooltip stays on its no-sources copy. This also blocks a mounted pin for the blocked-Handoff row, which task-32306 had to pin as a function instead.

- test_library_workspaces_mode_preserves_global_visibility_and_blocks_cross_workspace_handoff
- test_library_details_section_renders_grouped_headers_and_drops_policy_prose

**`Tests/UI/test_library_entry_compose_once.py` (2)**

- test_library_graduation_toast_is_not_repeated_by_reconcile_or_same_route_replace
- test_pending_conversation_open_cannot_overwrite_same_route_user_selection

**`Tests/UI/test_library_prompts_canvas.py` (~26)** — one was traced: `test_prompt_selection_clear_boundaries_and_invalid_row_fail_closed` asserts `app.notify.assert_not_called()` after a dirty rail-row switch, but that path has notified since `library_screen.py:21405` (`_notify_prompt_dirty_veto`) — the pin, not the product, is stale. The rest were recorded, not diagnosed:
  - test_library_prompt_pager_first_and_filter_failure_states[size1]
  - test_library_prompt_page_focus_survives_loading_recompose
  - test_prompt_selection_clear_boundaries_and_invalid_row_fail_closed
  - test_library_prompts_unmount_revokes_late_apply_before_workspace_shutdown
  - test_library_prompts_stale_search_cannot_restore_an_old_filter_caret
  - test_library_prompt_row_class_matches_notes_row_visual_parity
  - test_library_prompts_header_filter_empty_have_css_blocks
  - test_library_prompt_editor_field_css_blocks_match_notes_editor_parity
  - test_library_prompt_field_hint_css_block_matches_field_label_parity
  - test_library_prompts_import_row_css_blocks_match_filter_status_parity
  - test_library_prompt_bulk_delete_focus_and_refresh_are_exactly_once[selected_positions0-0-1]
  - test_library_prompt_bulk_delete_focus_and_refresh_are_exactly_once[selected_positions1-1-2]
  - test_library_prompt_bulk_delete_focus_and_refresh_are_exactly_once[selected_positions2--1--2]
  - test_library_prompt_bulk_delete_focus_and_refresh_are_exactly_once[selected_positions3-1-None]
  - test_library_prompt_delete_refreshes_applied_final_page_once_and_clamps
  - test_library_prompt_delete_refresh_failure_keeps_reconciled_page_read_only
  - test_library_prompt_undo_refreshes_applied_page_and_preserves_basket
  - test_library_prompt_import_blocks_undo_until_import_settles
  - test_cancelled_prompt_import_retains_writer_ownership_until_commit
  - test_library_prompt_conflict_save_as_new_replaces_source_history_identity
  - test_library_prompt_compatibility_editor_discard_returns_to_current_list
  - test_library_prompt_history_geometry_uses_only_the_outer_editor_scroll[dirty-size0]
  - test_library_prompt_history_geometry_uses_only_the_outer_editor_scroll[dirty-size1]
  - test_library_prompt_history_geometry_uses_only_the_outer_editor_scroll[dirty-size2]
  - test_library_prompt_history_geometry_uses_only_the_outer_editor_scroll[dirty-size3]
  - test_library_shell_create_prompt_save_creates_and_increments_count
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `Tests/UI/test_library_footer_focus.py` passes on dev, with its fake self carrying whatever the method under test now reads
- [ ] #2 `Tests/UI/test_post_release_workspaces_library_depth.py` passes on dev, with the seeded cross-workspace sources actually reaching the screen
- [ ] #3 `Tests/UI/test_library_entry_compose_once.py` passes on dev
- [ ] #4 `Tests/UI/test_library_prompts_canvas.py` passes on dev, and each stale pin's replacement asserts the behaviour that shipped rather than the one it was written against
- [ ] #5 Every fix says in one line whether the pin or the product was wrong, so a reader can tell a repair from a weakening
<!-- AC:END -->
