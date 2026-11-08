---
id: TASK-39165
title: "Twelve Console prompts-modal tests press, wait one pilot.pause() and query, so they fail under load as TASK-38977's two recipe tests did"
status: To Do
created_date: 2026-11-08 06:54
labels:
- follow-up
- tests
- console
- flaky
priority: low
references:
- Tests/UI/test_console_prompts_modal.py
dependencies:
- TASK-38977
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 12 of the 2026-10-23 burn-down (the ctests stream and its reviews, round-12 branch fu/r12-ctests). The pattern was checked by reading on the round-12 stack 9091820bf2, 2026-11-08. Tests are named by symbol, since round 12 changes the file.

**The precedent.** TASK-38977 (round 12) found two recipe tests that pressed a button after one `pilot.pause()`. They failed 1-4 of 160 runs at load ~40: `NoMatches` on a button the remount had not mounted yet, or a stale Save menu. It added `_mounted(root, selector)` and `_recipe_fill_settled(modal)` to the file, and moved those two tests onto them.

**The twelve.** Whole-file runs in the ctests stream, at load 60-90, failed 0-7 of these per run. All have the "press, one `pilot.pause()`, then query" shape:
- `test_prompts_guard_visible_cannot_be_bypassed[backdrop-recipe]`
- `test_recipe_save_confirms_library_destination_and_opens_saved_identity`
- `test_recipe_fill_snapshots_current_editor_definition[*]`
- `test_saved_recipe_ai_fill_snapshots_accepted_source_with_raw_identity`
- `test_auto_success_returns_one_apply_transaction_and_closes`
- `test_builtin_recipe_manual_guard_has_no_saved_source`
- `test_held_improve_activation_navigation_cancels_and_ignores_late_resolution[escape]`
- `test_improvement_system_read_permission_is_request_only[absent-recipe]`
- `test_leaving_saved_recipe_chooser_clears_selection_intent[close]`
- `test_recipe_manual_paths_work_without_provider_and_lane_defaults_are_safe`
- `test_recipe_retry_save_is_absent_from_compact_tab_order_until_failure`
- `test_stale_review_validation_cannot_mutate_or_commit_after_remount[success]`

**An example.** `test_recipe_fill_snapshots_current_editor_definition` presses the recipe button, its editor buttons and Fill with AI. Each press is followed by one or two `pilot.pause()` before `query_one(PromptBlockEditor)` and the request assertions.

**Lanes.** Four of the file's nodes are UI census entries (TASK-38977's). None of the twelve is laned.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each listed test waits on the end condition of each press it depends on (the mounted widget, the settled fill), using the file's existing helpers where they fit
- [ ] #2 The file passes 10 of 10 whole-file runs at `-n 4`, with load recorded, at least five of them at a one-minute load of 40 or more
- [ ] #3 A temporary edit that delays a remount turns a converted test red on its bounded wait, not on `NoMatches`
<!-- AC:END -->
