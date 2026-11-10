---
id: TASK-39165
title: "Twelve Console prompts-modal tests press, wait one pilot.pause() and query, so they fail under load as TASK-38977's two recipe tests did"
status: Done
assignee:
- '@claude'
created_date: 2026-11-08 06:54
updated_date: 2026-11-10 17:04
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
- [x] #1 Each listed test waits on the end condition of each press it depends on (the mounted widget, the settled fill), using the file's existing helpers where they fit
- [x] #2 The file passes 10 of 10 whole-file runs at `-n 4`, with load recorded, at least five of them at a one-minute load of 40 or more
- [x] #3 A temporary edit that delays a remount turns a converted test red on its bounded wait, not on `NoMatches`
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Run the file whole at `-n 4` on base to get a baseline under the current load.
2. For each of the twelve tests, replace "press, one `pilot.pause()`, query" with a bounded wait on that press's own end condition: `_mounted(root, selector)` for a remount's next widget, `_recipe_fill_settled(modal)` for Fill/Improve workers, and `until(...)` on the state a non-remounting press changes (guard display, `_recipe_selecting`, save calls, applies, results). Shared helpers (`_open_dirty_prompt_mode`) are fixed once for all callers.
3. Keep every assertion as it is; no sleep is lengthened.
4. AC#3: temporarily delay `_mount_recipe_editor`'s remount in the modal; show a converted test failing on its bounded wait ("... never mounted"), not `NoMatches`; revert.
5. AC#2: 10 whole-file runs at `-n 4`, recording the 1-minute load before and after each, at least five at load >= 40.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Test-only. Base ae9f9e8d21, 2026-11-10. Every listed test, and every other press-then-`pilot.pause()`-then-read site in the file, now waits on its own press's end condition. Each wait is bounded (10-15 s) and fails naming the state it waited for.

**Waits.**
- `_mounted(root, selector)` (TASK-38977's): the widget a press's remount mounts.
- New `_recipe_editor_mounted(root)`: the block editor and both lanes' cards. Each lane's cards compose inside its Collapsible a pump after the editor mounts, and one run read `NoMatches '#prompt-block-goal'` with the editor present.
- `_recipe_fill_settled(modal)`: any improvement worker (auto, review, Fill with AI). A second improvement first waits for its own request, because the first worker's finished state would satisfy the settle check at once.
- New `_editor_change_reached_modal(modal, editor)`: a block edit reaches the modal as a posted message, and Fill reads the modal's copy.
- New `_guard_offers_keep_editing(modal, what)`: the dirty guard shows at once but focuses Keep editing after a refresh.
- `until(...)` on what a press changes without a remount: applies, results, request and activation ids, a captured worker's `is_finished`, focus.
- The stale-review re-push now waits for the pop that `dismiss()` returned (`AwaitComplete.is_done`). One pause, or `not is_running`, let the re-push reuse the old instance, which never mounted again (lesson added to `lessons-textual.md`).
- A press that must do nothing (a disabled or refused duplicate) keeps one `pilot.pause()`. No assertion was changed or removed, and no sleep was lengthened. `_open_dirty_prompt_mode` is fixed once for its five callers.

**Why beyond the twelve.** After converting only the twelve, 10 whole-file runs passed 6 of 10 at load 54-106. The failures were same-shape tests: `test_recipe_fill_mounts_service_block_prompt_as_unsaved_prompt_review` (3 runs), `test_recipe_chooser_explains_each_starting_point_before_selection`, `test_held_improve_activation_disables_duplicate_resolution` and `test_normalized_browse_detail_and_update_use_source_id_not_composite_id` (each `NoMatches`), plus the first re-push wait (2). AC#2 is about the whole file, so every such site was converted.

**Red before (base, `-n 4`).** Whole file: 6 failed, 119 passed at load 30-75 (`test_improvement_system_read_permission_is_request_only[excluded-recipe]` and `[absent-recipe]`, `test_recipe_retry_save_is_absent_from_compact_tab_order_until_failure`, `test_recipe_fill_snapshots_current_editor_definition[outcome]`, `test_saved_recipe_ai_fill_snapshots_accepted_source_with_raw_identity`, `test_prompts_cancelling_improvement_dismissal_stays_in_transaction[escape]`). A second run gave 3 failed, 122 passed at load 73-58 (`test_recipe_fill_snapshots_current_editor_definition[blank]` and `[outcome]`, `test_builtin_recipe_manual_guard_has_no_saved_source`), each `NoMatches` on a recipe button after one pause.

**AC#2.** 10 whole-file runs at `-n 4`, two at a time: 10 of 10 (125 passed each). The 1-minute load read 75-93 at the start and end of every run. One more run after the guard wait became a helper: 125 passed, load 84-86.

**AC#3.** A temporary `await asyncio.sleep(12)` before `_mount_recipe_editor`'s body mount, at load ~91, failed five converted nodes on their bounded wait. Four failed with `AssertionError: #console-prompts-editor never mounted`, and the save test failed with `#prompt-editor-save-menu never mounted`. The nodes: `test_builtin_recipe_manual_guard_has_no_saved_source`, `test_recipe_fill_snapshots_current_editor_definition[blank]` and `[outcome]`, `test_recipe_save_confirms_library_destination_and_opens_saved_identity`, and `test_recipe_manual_paths_work_without_provider_and_lane_defaults_are_safe`. The `NoMatches` chained after each assertion is the product's delayed mount resuming during app shutdown. Reverted; `git diff` shows no product change.

**Files.** `Tests/UI/test_console_prompts_modal.py`, `backlog/docs/lessons-textual.md`.
<!-- SECTION:NOTES:END -->
