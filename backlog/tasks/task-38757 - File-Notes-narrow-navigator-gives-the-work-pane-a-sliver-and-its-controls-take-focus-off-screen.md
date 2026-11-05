---
id: TASK-38757
title: "File Notes narrow navigator gives the work pane a sliver and its controls take focus off screen"
status: Done
assignee:
- '@claude'
created_date: 2026-11-05 07:11
updated_date: 2026-11-05 22:06
labels:
- follow-up
- library
- file-notes
- ux
- accessibility
priority: medium
references:
- tldw_chatbook/Widgets/Library/library_file_notes_workspace.py
- tldw_chatbook/Utils/adaptive_reader_state.py
- Tests/UI/test_library_file_notes_workspace.py
- Docs/User_Guide/library/file-notes.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 7 of the 2026-10-23 follow-up burn-down. Checked on dev bb80905eb3, 2026-11-04.

Below 80 columns, the File Notes workspace switches to narrow navigation (`_apply_responsive_layout`, `Widgets/Library/library_file_notes_workspace.py:3536`). In the navigator view it resolves the shared adaptive reader layout with `priority="items"` (:3561-3571). The items pane and two 5-cell grips are seated first, and the work pane gets what is left.

A probe on this base called `resolve_adaptive_reader_layout(width, AdaptiveReaderLayoutPreferences(), AdaptiveReaderLayoutProfile(), priority="items")`. With items at 32 cells, the work pane got 37 cells at 79 columns, 18 at 60, 8 at 50 and 2 at 44, and none at 42 and below. At 40 the items pane itself dropped to 30.

That leftover pane is still laid out, and its controls stay in the focus chain. During the File Notes narrow-width work in PR #3403 (not yet on dev), at 40 columns, "More file actions" (`#file-notes-maintenance-toggle`) took focus at x=41, off screen, and the editor pane painted 1 cell wide at x=40. #3403 fixes the width-0 crash in that layout (its one-row buttons now truncate instead of wrapping). It left hiding the sliver alone, because that is a change to the shared resolver.

So between 40 and 79 columns the navigator shows a work pane too thin to use, and Tab can land on a control the user cannot see.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 In the narrow navigator view (40-79 columns), the work pane is either hidden or wide enough to use; no sliver of a few cells is shown
- [x] #2 At every width from 40 to 79, Tab from the navigator never focuses a widget outside the visible region
- [x] #3 In the narrow editor view (after opening a file, and after Back), the work pane and its controls still show and take focus as before
- [x] #4 A width-sweep test over 40-79, in the navigator and editor views, pins #1-#3 and is red before the change
- [x] #5 If the narrow layout changes, the File Notes User Guide page describes it
<!-- AC:END -->

## Implementation Plan

1. Write a width-sweep test in `Tests/UI/test_library_file_notes_workspace.py` over 40-79 in three cases: the navigator with no file, the editor view after opening a file, and the navigator after Back followed by a reopen. At each width it checks that the navigator's work pane is hidden or wide enough to use, that every widget in `screen.focus_chain` has visible cells inside its clip, and that Tab from the tree lands on a visible widget. In the editor view it checks that the work pane shows at `width - 10` and that Back is in the focus chain and takes focus. Show it red on base.
2. Fix this in the workspace, not in the shared resolver. Both layout paths (the standalone resolve and the Library screen's) go through `sync_reader_layout`. There, when the narrow navigator would leave the work pane a sliver, give its columns to the Files pane and hide the work pane in `_sync_body_panes`.
3. Re-run the File Notes covering tests. Update the User Guide's narrow-layout description, and append the new nodes at the census tail.

## Implementation Notes

- The fix is in the workspace, not in the shared resolver. Both layout paths, the standalone resolve and the Library screen's `library_notes_controller`, deliver their layout through `sync_reader_layout`. When the narrow navigator view would leave the work pane under `_NARROW_WORK_MIN_WIDTH` (18) cells, that method gives the pane's columns to the Files pane. `_sync_body_panes` then hides the work pane, which takes its controls out of the focus chain. The editor view is unchanged.
- Why 18: it is what the navigator leaves the work pane at 60 columns. My first threshold was 30, the narrow editor's own width at 40 columns. It broke 8 pinned 60x20 tests, which require the work pane's save and recovery status to be painted whole at 60x20: `test_file_notes_authority_is_painted_and_contained_at_60x20_shell`, `test_file_notes_merged_recovery_authority_paints_at_60x20_shell` (2 cases), `test_file_notes_combined_non_ready_authority_matrix_paints_at_60x20` (4) and `test_file_notes_authority_copy_is_complete_and_bounded[size2]`. With 18, the navigator shows the work pane from 60 columns (18-37 cells) and hides it from 40 to 59 (it was 0-17 cells). Trade-off: below 60 columns, Manage and Session Git are reached by opening a file first.
- The sweep also found the navigator's **New** button invisible at every width. The header's title `Static` took the whole row, and Button's 16-cell `min-width` put New wholly outside the row's clip. It was still in the focus chain, so Tab landed on it unseen (AC#2). The title is now `width: 1fr`, and New joins the compact button group.
- Test: `test_file_notes_narrow_views_lay_out_at_every_width_40_to_79[navigator|editor|after-back]`. At each width it checks:
  - the navigator's work pane is hidden or at least 18 cells;
  - every widget in `screen.focus_chain` has visible cells inside its compositor clip;
  - Tab from the tree lands on a visible widget;
  - in the editor view, the work pane shows at `width - 10`, and Back is in the focus chain and takes focus. After Back, a reopened file is checked at 79 and 40.
- Red on base: `[navigator]` and `[after-back]` fail at 40 with the work pane 1 cell wide (`(40, 1)`), and `[editor]` passes, as AC#3 expects. With the layout fix but not the New fix, `[navigator]` fails at 40 with `('file-notes-new', 33, 16)`. Green after both: 6-12 s per case at load 23 on 18 CPUs.
- Covering run: `test_library_file_notes_workspace.py`, `_git.py`, `_git_push.py`, `_delete_safety.py`, `_history.py`, `test_non_obscuring_focus_contract.py` and `test_widget_css_consolidation.py` (`-n 4`): 625 passed, 6 failed. The 6 are TASK-32185's known reds, and they are red on the base workspace source with the same command. In the wider Library run, the other failures were the local `RecoveryRequired` trap, or red on base as well (`test_the_trust_dialog_never_breaks_the_path_it_asks_consent_for`, `test_library_modal_inventory_*`).
- The three nodes are appended at the census tail. The `library_file_notes_workspace.py` module-size row goes 9674 -> 9706 under the owner's waiver. The User Guide's Layout tour now describes the narrow layout. A lesson is added to `lessons-textual.md`.
- Files: `tldw_chatbook/Widgets/Library/library_file_notes_workspace.py`, `Tests/UI/test_library_file_notes_workspace.py`, `Docs/User_Guide/library/file-notes.md`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`, `Tests/Architecture/test_module_size_ratchet.py`, `backlog/docs/lessons-textual.md`.

