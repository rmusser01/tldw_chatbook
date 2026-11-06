---
id: TASK-38983
title: "File Notes: during a failed or conflicted save, Choose folder is offered but silently refused, and only typing retries the save"
status: Done
assignee:
- '@claude'
created_date: 2026-11-06 01:42
updated_date: 2026-11-06 03:58
labels:
- follow-up
- library
- file-notes
- ux
priority: medium
references:
- tldw_chatbook/Widgets/Library/library_file_notes_workspace.py
- Tests/UI/test_library_file_notes_workspace.py
- Docs/User_Guide/library/file-notes.md
dependencies:
- TASK-38757
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down. Checked on dev 29592ecc9b and on the TASK-38757 branch, which merges first, 2026-11-05. Lines refer to that branch's `tldw_chatbook/Widgets/Library/library_file_notes_workspace.py`.

**Probe on dev (load 6.1, 120x40).**
- Steps: open a file, edit it, rename its folder away, let the autosave run, then a poll.
- The save state becomes `error`, and the status reads "Unavailable — the folder cannot be reached; your draft is preserved. Next: Choose folder."
- `set_root(<another valid folder>)` then returns False. The folder does not change, the action status stays empty, and the folder row still reads "Offline · Local folder: notes". The user is told nothing.

**Why Choose folder is refused.**
- The status resolver offers "Choose folder" for `unavailable`, and "Save Copy" for a conflict or a failed save (`:209-227`).
- `set_root` returns early at `if not await self.flush_pending_work(): return False` (`:6548-6549`), before it reports anything.
- `flush_pending_work` refuses whenever the save is in `conflict` or `error` (`:7111-7112`).
- `_change_root_with_deadline` reports nothing for a quick False.

**No Retry.**
- The only Retry copy belongs to Git (`:5646`, `:5707`, `:5734`), and there is no ctrl+s.
- `_start_autosave` runs only from `dirty` (`:6971-6979`), and only typing sets that state (`_editor_changed`, `:7193-7206`).
- In a conflict, typing does not retry either (`:7201`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The "Next:" action shown during a save error or conflict succeeds when taken, or the hint names a step that does
- [x] #2 A refused folder change says why on the folder row instead of silently doing nothing; red before: the dev probe (False returned, folder row and action status unchanged)
- [x] #3 A failed save offers an explicit Retry, as a control or a key, that re-runs the save without editing the text; a conflict keeps its Save Copy and Reload choices
- [x] #4 Tests pin #1-#3 and are red on dev
<!-- AC:END -->

## Implementation Plan

1. Probe test first (workspace harness, 120x40): open, edit, rename the folder away, autosave, poll. Assert the hint names a step that works, `set_root(other)` says why on the folder row, and after the folder is back a Retry control saves the unchanged draft. Show it red on base.
2. Status resolver: while the linked folder is offline and the draft is not saved (dirty, saving, failed, or conflict), "Next:" names "Reconnect the folder" instead of "Choose folder" or "Save Copy", which both fail there.
3. `set_root`: when the leave guard refuses because the draft is not saved, report the reason on the folder row (`_report_root_change_reason`); clear that reason once the draft saves.
4. Add a "Retry save" recovery button, shown only in the failed-save state, that re-runs the save from the current text and moves focus to the editor. A conflict keeps Compare, Resolve, Reload and Save Copy, without Retry.
5. Update the pinned copy and contextual-action sets, the User Guide, and run the covering File Notes tests.

## Implementation Notes

- **AC#1, the "Next:" step.** While the linked folder is offline and the draft is not saved (dirty, saving, failed, or a conflict), `resolve_file_note_status_channels` now names "Reconnect the folder". "Choose folder" is refused there by the leave guard (it would drop the draft), and "Save Copy" writes into the same unreachable folder. Online, a failed save and a conflict keep "Save Copy", which succeeds; with no unsaved draft, offline keeps "Choose folder", which `set_root` accepts.
- **AC#2, the refusal.** `set_root` now reports `ROOT_CHANGE_DRAFT_COPY` ("Folder not changed · your draft is not saved yet.") on the folder row (`_report_root_change_reason`) when the leave guard refuses because the save state is `error` or `conflict`. The reason clears when the draft saves (`_set_save_state("saved")`), on the next folder change, or when a file opens, which are the existing reason rules.
- **AC#3, Retry.** A **Retry save** button (`#file-notes-recovery-retry`) joins the contextual recovery row, shown only in `error` and disabled like Reload during a file or Git operation. It re-runs the save from the unchanged editor text (`dirty`, then `_start_autosave`). It moves focus to the editor first, synchronously, because it hides itself once the state leaves `error`. A deferred `focus()` lost to the hide's fallback, which focused New. A conflict still shows Compare, Resolve conflict, Reload from disk and Save Copy, without Retry. It is a control, not a key, since the guide says Folder files does not use Ctrl+S.
- **Test.** `test_file_notes_failed_save_names_a_working_step_and_retries_unchanged` (120x40): it renames the folder away under a dirty draft, autosaves and polls. It checks the "Reconnect the folder" hint, then that `set_root(other)` is refused with the folder-row reason. It renames the folder back and polls (the hint returns to Save Copy), presses Enter on Retry save, and checks the disk text and editor focus. Finally the folder row is Linked again and `set_root(other)` succeeds.
  - Red on the base source: `'Next: Reconnect the folder.' in '... Next: Choose folder.'`.
  - Negative controls: without the reason line, `'Offline · Lo...folder: notes' == 'Folder not c...ot saved yet.'`; without the Retry body, "Retry did not save".
  - Green: 1.4-1.5 s.
- **Changed pins.** Each pinned the old behaviour.
  - `test_file_notes_primary_actions_follow_editor_state`: the `error` contextual set gains `file-notes-recovery-retry`.
  - `test_file_notes_combined_non_ready_authority_matrix_paints_at_60x20`: offline now expects "Next: Reconnect the folder." for both error and conflict. All 4 cases still paint whole at 60x20.
  - `test_high_stakes_file_notes_states_keep_explicit_labels_and_classes`: offline + conflict now expects "Next: Reconnect the folder.".
- **Covering runs** (`-n 4`, load 17-26 on 18 CPUs).
  - `test_library_file_notes_workspace.py`: 190 passed, 7 failed before the high-stakes pin was updated (that pin passes now). The other 6 are TASK-32185's known reds, red with the same command on the exact-base worktree `fu-base-r10`.
  - The other File Notes files (delete_safety, git, git_push, history, retention, search_debounce, notes_reader, riders, w4, w5 folder layout and import preview, wave, crit8 waits and riders, non_obscuring_focus, widget_css, Backup_Recovery notes and unsaved editors, the poll gating) plus the three changed pins: 665 passed, 55 failed. These are exactly the 55 that fail on `fu-base-r10` with the same command, mostly the local `RecoveryRequired` trap.
- The module-size row goes 9802 -> 9835 under the owner's waiver. The node is appended at the census tail, floor 1701. The User Guide describes Retry save, the offline next step, and the refused folder change.
- Files: `tldw_chatbook/Widgets/Library/library_file_notes_workspace.py`, `Tests/UI/test_library_file_notes_workspace.py`, `Tests/Architecture/test_module_size_ratchet.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`, `Docs/User_Guide/library/file-notes.md`.

### Review fixes (r10l-rev1)

- **Test gap closed.** `test_file_notes_retry_after_the_file_changed_on_disk_ends_in_conflict`: a save fails, the file then changes on disk, and Retry save ends in Conflict. The disk keeps "changed on disk", the editor keeps the draft, and the conflict shows Save Copy and Reload without Retry.
- **Evidence.** The test passes (4.1 s), since Retry already goes through the guarded save. Negative control: a Retry that adopts the disk hash before saving overwrote the file, and the test failed with "Retry did not end in Conflict (saved)". The source was restored. The node is appended at the census tail.

