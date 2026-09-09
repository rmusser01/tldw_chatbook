---
id: TASK-32136
title: >-
  Library Notes Folder files is a mode of Notes: keep the rail and the source
  strip, and offer the configured sync folder — user decision 2026-09-09
status: Done
assignee: []
created_date: '2026-09-08 21:39'
updated_date: '2026-09-09 06:35'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - file-notes
  - layout
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The user decided Folder files is a mode of Notes, not a separate screen. Today switching to it replaces the whole canvas: the Library rail, the Notes list and the 'Library notes | Folder files' strip itself disappear, leaving a '‹ Library / Notes' cue; the empty state ('Choose a notes folder.') does not explain the mode or offer the already-configured `[notes] sync_directory`. Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-09T04-24-50Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev c4a7b1911f, 2026-09-08/09, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile with a 71-file Obsidian-style vault; parent re-tested every disagreement on a third profile).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 At wide sizes the Library rail and the source strip stay visible inside Folder files, before a folder is linked as well as after (the qualifier this AC carried was removed by task-32173)
- [x] #2 The empty state explains in one line what Folder files does and offers the configured sync folder when one is set
- [x] #3 The file-notes guide's layout tour matches
- [x] #4 Covered by a compose test at 235x52
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Failing compose test at 235x52: Folder files keeps the rail and the Library notes | Folder files strip.
2. Stop treating wide Folder files as a focused task that collapses the source strip (compose + _sync_library_notes_source_controls).
3. Empty state: one-line explanation plus a Use <folder> button for [notes] sync_directory.
4. Docs layout tour + stamps.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Folder files is a MODE of Notes (user decision 2026-09-09), so the strip you switch in with is the strip you switch back with.

- Wide Folder files no longer counts as a *focused task* in the source strip: one added term (`not self._file_notes_active()`) at the two places that compute `wide_focused_task` -- `library_screen.py`'s compose and `library_notes_controller._sync_library_notes_source_controls`. Compact is unchanged (there the strip really is a stage). **Library notes** already routes through the guarded `_return_to_library_database_notes`, so this is a real way back, not a bypass.
- The Library rail turned out to be already mounted and visible inside Folder files (`#library-file-notes-rail`, 34 cells at 235x52); the critique's 'the rail goes with it' was about `#library-rail`, the *database* rail, which is correctly hidden. The new compose test pins the file-notes rail rather than changing anything.
- Empty state: a wrapping `#file-notes-empty-purpose` line ('Folder files edits Markdown files in a folder on disk, in place. Nothing is copied into the Library.') plus a `Use <folder>` button when `[notes] sync_directory` names a real directory. Deviation from the brief: the sentence is its own line rather than replacing the folder-row status -- that status is a single nowrap `width: auto` line (task-2850 keeps the prompt adjacent to its button), and a 100-character sentence there pushes 'Choose folder…' off a 100- or 60-column row.

Three existing tests pinned TASK-19602's hidden wide strip and were updated to this superseding decision (two renamed from `..._task_return_...` to `..._source_switch_...`, since the control under test moved).

Live at 235x52 on the power profile: the strip stays inside Folder files, 'Use file_notes' linked the configured folder, and the rail is beside the vault tree (captures 02, 03, 09).

Files: tldw_chatbook/UI/Screens/library_screen.py, tldw_chatbook/UI/Library_Modules/library_notes_controller.py, tldw_chatbook/Widgets/Library/library_file_notes_workspace.py, Tests/UI/test_library_notes_wave_file_notes.py (new), Tests/UI/test_library_file_notes_workspace.py, Docs/User_Guide/library/notes.md, Docs/User_Guide/library/file-notes.md.
FINAL WHOLE-BRANCH REVIEW (2026-09-09), I2 -- AC#1 re-marked `[~]`. Controller ruling: the rail this task pins is the workspace's OWN rail, mounted inside `#file-notes-body`, and `library_file_notes_workspace.py`'s root-status render sets `body.display = self._root is not None`. So before a folder is linked there is no rail at all -- only half of AC#1 holds unconditionally. The source strip DOES survive the unlinked state, so the user is never trapped in Folder files, which is why this is a qualifier and not a reopened defect.

Renumbering-free rider (follow-up work, no new task id): give the pre-link Folder files empty state the workspace rail, so the mode reads as a mode from the first frame rather than as a full-width onboarding step. Whoever picks it up should start at the `body.display` gate above -- the rail is a child of the body that gate hides, so it needs either its own display rule or a rail that lives outside `#file-notes-body`.

DONE by task-32173 (2026-09-09): the `body.display` gate is gone. Hiding the body left the reader shell with zero width, so its adaptive layout never resolved and `library_open` was False even at 235 columns -- the rail could not have been un-hidden on its own. The body now stays mounted and `_sync_body_panes` gates only `items`, `work` and the two grips; the resolver owns the rail and already closes it below the Library compact breakpoint, so compact is unchanged. AC#1 re-ticked `[x]`.

`Docs/User_Guide/library/file-notes.md`'s task-32136 stamp on this branch now carries the same qualified wording the docs group applied on the integration branch, so the two do not disagree. No code change.
<!-- SECTION:NOTES:END -->
