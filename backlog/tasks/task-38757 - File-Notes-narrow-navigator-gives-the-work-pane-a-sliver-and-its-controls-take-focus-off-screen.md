---
id: TASK-38757
title: "File Notes narrow navigator gives the work pane a sliver and its controls take focus off screen"
status: Done
assignee:
- '@claude'
created_date: 2026-11-05 07:11
updated_date: 2026-11-06 00:15
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
- [x] #6 When the narrow navigator hides the work pane, anything that must be seen there brings it forward on the real Library screen too: a selected deleted note (Restore visible and focusable), New's path field, and an open or save that fails or is refused (its message and recovery painted). Cancelling a New that brought the pane forward returns to the navigator with New focused, and a standing save failure or conflict stays in view through a resize. An explicit navigator choice (‹ Files, or the Folder files grip in the narrow editor) always reaches the navigator, with no work-pane sliver
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

### Review fixes (r8f-rev1)

- **A deleted note selected at 40-59 columns showed nothing on the Library screen.** `select_deleted` switched to the editor view without asking the screen, which owns the external shell's layout, to re-resolve. Every narrow-view switch now goes through `_set_narrow_view`: open, deleted-note select, Back and Manage. When the shell is external, it re-syncs the body panes and posts `AdaptiveReaderShellResized`. The one remaining direct assignment is the enter-narrow transition inside `_apply_responsive_layout`; it now posts the message too.
- **Failures showed nowhere while the work pane was hidden.** `_bring_work_forward` switches to the editor view when the navigator hides the work pane. It runs in three places:
  - when the save state enters `error` or `conflict` (an autosave that fails after Back);
  - in `open_path`, on a flush refusal, a refused path transition, or "Open failed";
  - when New opens its path form, which lives in the work pane. I found this one myself: New is reachable only from the navigator, and its form and focused field were hidden.
- **New at wide sizes.** `test_file_notes_delete_is_spatially_separated_from_new` now measures New's cells inside its clip with `_seen_cells`, the helper the sweep now shares, and gains 211x44 and 235x52.
- **Red and green evidence.**
  - New test `test_file_notes_hidden_work_pane_comes_forward_on_the_library_screen[deleted|open-error|save-error|new]` runs the real Library screen at 50x24. All four cases are red on `d0f946520c`, each with "... did not come forward", and green after (5-6 s each at load 30).
  - The strengthened New check is red on the pre-fix source `29592ecc9b` at 120, 211 and 235 (`assert 0 == 16`), and green now.
- Covering run after the fixes, `-n 4`, over the workspace, git, git_push, delete_safety, history, retention and search_debounce files, the riders, w4 and wave File Notes files, `test_non_obscuring_focus_contract.py` and `test_widget_css_consolidation.py`: 665 passed, 6 failed. The 6 are the same TASK-32185 known reds as before.
- The four new nodes are appended at the census tail. The module-size row goes 9706 -> 9730. The User Guide bullet now says "until you open or select a file", and that New or a failed or refused open or save brings the work area forward.

### Re-review fixes (r8f-rev2)

- **Cancelling New left no focus at 40-59 columns.** `_open_path_task` records whether bringing the pane forward switched the view. A cancelled close (the Cancel button, the Library screen's Escape through `cancel_path_task`, or `_open_path_task("none")`) then calls `_set_narrow_view("navigator")`. After the refresh it focuses the opener, through `_focus_once_shown`, which retries until the screen has re-laid out its shell. A completed New opens the new file and stays in the editor (`cancelled` is False on that path).
- **A resize could hide a standing failure.** `sync_reader_layout` no longer hides the work pane while the save state is `error` or `conflict`. It switches to the editor view instead, which re-resolves the layout. (rev3 reverses the side effect this had on ‹ Files; see below.)
- **Optional recovery focus: not done.** Picking the recovery control (Save Copy or Reload) depends on the state and the mode, and each choice risks focusing a control that is hidden. The bring-forward leaves focus where it was.
- **Tests.** The Library-screen test gains `[new-create]`, `[new-cancel]`, `[new-escape]`, `[new-cancel-open]` and `[resize-failure]`. `[resize-failure]` shows "Save failed" beside the navigator at 65 columns, resizes to 55, and also checks that Back keeps it in view; it passed 3 of 3 runs serially. All 9 cases pass in 15 s with `-n 4`.
- **Red evidence on `85fb887559`.** `[new-cancel]`, `[new-escape]` and `[new-cancel-open]` fail with "cancelling New did not return to the navigator", and `[resize-failure]` fails with "'Save failed' did not come forward". The four rev1 cases pass there.
- Covering run after these fixes (the same files as in rev1, `-n 4`): 670 passed, 6 failed. The 6 are TASK-32185's known reds, and they fail the same way on the exact-base worktree `fu-base29592`.
- The five new nodes are appended at the census tail, with literal parametrize ids so the census checker can resolve them. The module-size row goes 9730 -> 9779. The User Guide bullet adds a conflict with the file on disk, a cancelled New, and a standing failure staying in view.

### Re-review fixes (r8f-rev3)

- **An explicit ‹ Files always wins** (the coordinator's ruling). The rev2 guard made ‹ Files do nothing during a standing failure below 60 columns. Two states had no way out at all: a folder renamed away, and an unreadable file in a read-only folder. A new `_navigator_chosen` flag fixes this. Back sets it, and the Files grip (below) sets it too. It is cleared when the save state changes or the view switches to the editor. The `sync_reader_layout` guard keeps the work pane in view only while the flag is clear, so it now acts only on the automatic paths: a failure starting, and a resize. During the failure, opening another file is still refused, and that refusal brings the pane forward again (`_set_narrow_view("editor")` clears the flag), so the failure is never hidden for good.
- **The Files grip reopened Files beside an 8-cell work sliver.** On the Library screen at 50x24 in the narrow editor view, the items grip resolves `priority="items"`, which gave items 32 cells and work 8. `sync_reader_layout` now treats "narrow, items open, work under 18 cells" while the view says editor as a navigator choice: it switches the view, sets the flag and hides Back. Then the AC#1 hide applies.
- **Tests.** `[resize-failure]` now checks that ‹ Files reaches the navigator after the resize keeps the failure in view, and that opening `gone.md` is refused and brings "Save failed" forward again. A new `test_file_notes_narrow_navigator_choice_wins_on_the_library_screen[offline-back|grip-from-editor]` covers the other two cases:
  - `offline-back`: rename the folder away under a dirty draft, run the autosave until it fails (error or conflict), press ‹ Files.
  - `grip-from-editor`: press the items grip from the narrow editor.

  Both assert that the navigator takes the stage (work pane hidden, items `50 - 10` cells) and that no focusable widget in the workspace is outside its clip. The setup of both Library-screen tests moved into a shared `_library_folder_files` context manager.
- **Red on `3331c59fc5`:**
  - `[resize-failure]` fails with "Back did not show the navigator (work pane shown: False)".
  - `[offline-back]` and `[grip-from-editor]` fail with "the navigator did not take the narrow stage".

  Green after: all 11 Library-screen cases, 2-7 s each at load 9.
- Covering run (the same files, `-n 4`): 672 passed, 6 failed. The 6 are the same TASK-32185 known reds, red on `fu-base29592`.
- The two new nodes are appended at the census tail. The module-size row goes 9779 -> 9802. In the User Guide, "even after a resize or ‹ Files" becomes "a resize keeps it in view; ‹ Files still returns to the navigator, and opening another file is refused and brings the failure back". It also says the grip gives the navigator the whole width.

