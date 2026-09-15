---
id: TASK-32621
title: >-
  Library Notes: Folder files asserts Saved over no file selected and silently
  drops non-note entries from its tree
status: Done
assignee: []
created_date: '2026-09-15 06:43'
labels:
  - library
  - notes
  - critique-4
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 (dev 77eb2601a6), assessor A section 11 and assessor B section 6f, persona Riley, Obsidian workflow.

What happened. With no file selected, the Folder-files right pane shows the chip 'Saved' and a fully drawn, apparently editable body box beside the words 'No file selected' (A cap 29) -- a save state asserted over nothing, on the one surface whose entire promise is that it edits the real file. And the tree silently omits notes.csv, meta.yaml, a Canvas folder and an attachments folder, with no line explaining what is shown (A cap 29; B confirms the same omissions at cap 35 and reads them as correct behaviour -- which is the point: the behaviour is right and unexplained).

Related asymmetry worth stating in the same place: lasting sync silently ignores the same .csv and .yaml sources that Import once at least reports under Failed and Skipped, and the sync review's '0 need attention' hides it (A section 11).

Cause PROVEN by capture.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 With no file selected the right pane shows no save state and no editable body
- [x] #2 The Folder-files tree states what it lists, so a missing file is explained rather than silently absent
- [x] #3 The two vault-reading paths agree on whether an unreadable source is reported or ignored, or each says which it does
<!-- AC:END -->

## Implementation Plan

1. Find the producer of the "Saved" chip with nothing open.
2. Hide the editable body when there is no file to edit.
3. State what the tree lists, and what each vault-reading path reads.

## Implementation Notes

**Seven tests pinned the defect.** `test_library_file_notes_workspace.py` had
fourteen failing cases across seven functions asserting "Saved" as the content
channel with NO file open -- including one that asserted it on the line before
it opened its first file. Their subject is the Git channel and the
"no cross-channel masking" contract, which is preserved; only the false content
claim moved. Baseline established first: dev @3b26c66ce0 is **8 failed, 154
passed** on that file, and every one of those eight (three
`test_notes_authority_switch_*`, `test_notes_authority_round_trip_*` x2,
`test_wide_files_task_return_*`, `test_high_stakes_..._legible_in_shipped_
themes` x2) is red without this branch. The fourteen new ones are the copy
change and nothing else -- checked by disabling the editor-visibility half
alone, which changed none of them.

**Copy, second pass.** The first version was "No file open. Next: Choose a file
in the tree." Two things argued it down to "No file open.": at 60x20 the
sentence is three rows of a two-row box, and every other safe action on this
pane names a CONTROL ("Save Copy", "Choose folder", "Open Manage") where this
one named a gesture. The tree is the only thing to act on and it is already on
screen.

**AC#1 -- the producer.** `resolve_file_note_status_channels` is a chain of
`elif`s ending in `else: content = "Saved"`, and with no file open every
save-state input is False -- so the fallback asserted a save for a file that
does not exist. A `file_open` input (defaulting True, so every existing pin
stands) gives that state its own branch: "No file open. Next: Choose a file in
the tree". The editable body goes with it -- `_sync_editor_visibility`, called
from `_update_controls`, hides the retained editor when nothing is open
(hidden, not unmounted: every caller queries `#file-notes-editor`
unconditionally) and drops focus if it was there, since a hidden widget keeps
focus in Textual.

**AC#2 -- the tree says what it lists.** One line under the pane heading:
"Lists .md, .markdown, .txt and .text. Other files stay on disk." Both
assessors were right that the omission is correct behaviour; it was the only
thing on the pane not saying so.

**AC#3 -- each path says which it does.** Neither path changes what it does.
The lasting-sync review gains its own scope line under the summary ("Syncs
.md, .markdown and .txt only; other files are left alone"), so "0 need
attention" no longer covers sources Import once reports under Failed and
Skipped.

**Three different sets, deliberately.** Folder files reads
`SUPPORTED_EXTENSIONS` (.md/.markdown/.txt/.text), lasting sync reads
`_SYNC_FILE_EXTENSIONS` (.md/.markdown/.txt -- NOT .text), and Import once
reads `SUPPORTED_NOTE_EXTENSIONS` (nine, including .csv/.yaml). Each sentence
names its own set and each pin reads that set from its own constant rather
than retyping it, so a change to any of them fails the pin instead of rotting
the copy.

**Owned files touched (flagged for the merge check):**
`Widgets/Library/library_file_notes_workspace.py` -- a Static in
`_build_reader_items_pane`, one CSS rule, a new `_sync_editor_visibility`, and
the `file_open` argument; `library_notes_add_from_files_canvas.py` -- one
Static in `_compose_phase`'s review branch.

**Files.** `Widgets/Library/library_file_notes_workspace.py`,
`Widgets/Library/library_notes_add_from_files_canvas.py`,
`Tests/UI/test_library_notes_w5_import_preview.py`,
`Docs/User_Guide/library/file-notes.md`,
`Docs/User_Guide/library/notes.md`.
