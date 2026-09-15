---
id: TASK-32611
title: >-
  Library Notes: three folder pickers, one labelled File name while it can only
  pick a folder
status: Done
assignee: []
created_date: '2026-09-15 06:39'
updated_date: '2026-09-15 19:16'
labels:
  - library
  - notes
  - critique-4
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 (dev 77eb2601a6), assessor A P2 and assessor B D11, every persona, all three journeys. Heuristic 4 setter (2 -> 1).

What happened. One sub-screen pushes three unrelated folder dialogs.
- Import once: title 'Import once (files or one folder)', field 'File name or path' (and on a second visit the label flips to 'Folder path:'), buttons Open / Select folder / Cancel, arrives focused (A caps 18, 19; B caps 19, 20).
- Folder files: title 'Choose File Notes Folder' (Title Case, and 'File Notes' is internal jargon for a surface the UI calls Folder files), field 'Folder path', pre-filled, buttons Select / Cancel, arrives UNFOCUSED (A caps 27, 28).
- Keep a folder synced: title 'Choose a folder to keep synced', folder-only, field labelled 'File name' with placeholder 'File name or path' (A cap 47, B cap 30).
Hint text differs too ('Select folder to use this folder' vs 'Select to use this folder'), and all three default their listing to 'Discovery order', so a vault renders as Reading, scratch.txt, Inbox, Archive, Projects, Daily, Canvas, README.md, notes.csv, People, Templates, Ideas.md, meta.yaml, attachments -- files and folders interleaved, unsorted (A cap 19).

Cause PROVEN by the captures. The focus half of the Folder-files dialog is filed separately as its own blocker. Adjacent open rider: 32580 (a click-fill that does not select the filename it fills).

Riley note: the picker reports 'Loaded · 17 entries' while showing 14 plus '..' -- the three hidden dot-entries are counted but not shown (B cap 20, D13). Filed with the import nits, mentioned here because it is the same component.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 One picker component serves all three doors, with a mode flag for files-plus-folder versus folder-only
- [x] #2 Folder-only mode labels its field for a folder and offers only the buttons that can act on one
- [x] #3 All three doors arrive with the same focus, the same hint wording and the same button grammar
- [x] #4 The default listing order is folders-first, name-ascending, with Discovery order kept as an option
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace all three doors: Import once + Keep a folder synced are FileOpen(offer_select_folder=True); Folder files is SelectDirectory.
2. AC#1/#2: switch 'Keep a folder synced' to SelectDirectory -- it can only return a folder (its callback already drops non-dirs), so the folder-only mode of the one picker family is the right component. RETURNS_A_FOLDER stays the single mode flag (task-32606); no second source of truth.
3. AC#3: unify the hint on FileSystemPickerScreen._hint_text keyed on RETURNS_A_FOLDER and delete both overrides; give folder-only dialogs the same 'Select folder' confirm label via a SELECT_BUTTON_DEFAULT class attr.
4. AC#4: add a 'folders' sort key (folders first, name-ascending) and make it the default listing order for every folder-returning dialog; 'Discovery order' stays in the menu.
5. Pin RED->GREEN in Tests/UI, run the touched files plus the picker suites, both-sides name-set compare against origin/dev.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**AC#1/#2 -- the defect was the CALLER, not the picker.** "Keep a folder
synced" pushed `FileOpen(offer_select_folder=True)`, the files-AND-folder
dialog, while its own callback has always dropped anything that is not a
directory (`if path is None or not path.is_dir(): return`). So the field was
labelled "File name", the placeholder read "File name or path", and every file
in the folder was listed as though pickable -- under the title "Choose a
folder to keep synced". It now pushes `SelectDirectory`
(`library_notes_controller.handle_library_notes_lasting_folder_requested`),
the folder-only mode of the same family: "Folder path:", pre-filled and
selected, `show_files = False`, and only the two buttons that can act on a
folder. No picker code was needed for this half; the door was asking the wrong
question.

The "one picker component with a mode flag" the AC asks for already exists and
is deliberately NOT duplicated: `FileSystemPickerScreen` is the component and
`RETURNS_A_FOLDER` (task-32606) is the flag -- a class attribute on the two
directory-only dialogs, answered per instance by `FileOpen` from its existing
`offer_select_folder`. Everything below keys on that one fact rather than
adding a second one beside it.

**AC#3 -- one hint and one verb, both from the base.**
`FileOpen._hint_text` and `SelectDirectory._hint_text` are deleted;
`FileSystemPickerScreen._hint_text` returns
`Enter Open  ·  Select folder to use this folder` whenever `RETURNS_A_FOLDER`
and `""` otherwise, so a file-only picker is untouched. `FOLDER_CONFIRM_LABEL`
is the single literal behind both that sentence and the "Select folder" button
`compose` adds on a files-and-folder dialog; `SELECT_BUTTON_DEFAULT` (a class
attribute, `"Select folder"` on `SelectDirectory`) is the folder-only half, so
the button that commits a folder is called the same thing on all three doors.
A caller passing an explicit `select_button` still wins -- the character-import
and eval pickers keep "Import"/"Choose card". Focus was already identical
across the three after task-32606.

**AC#4 -- folders-first is the default only where it is the question.** New
sort key `"folders"` ("Folders first" in the menu, added to
`validate_file_picker_sort_key`). `project_records` sorts by name and then
applies a SECOND, stable sort on `not is_directory` -- folding the flag into
the name key instead would make "Descending" put files first, whereas
"Folders first" has to keep meaning folders-first in both directions.
`_default_listing_sort()` returns it for a `RETURNS_A_FOLDER` dialog and
`"discovery"` for everything else, read by `_listing_controls` (the Select's
initial `value=` and the direction control's `disabled=`) and by `on_mount`
(the navigation's `sort_key`). `on_mount` is the one place both picker
families are covered: Textual dispatches it to every class in the MRO, so
`EnhancedFileDialog` -- which builds its own navigation in its own `compose`
-- needs no copy. "Discovery order" stays on the menu, and a file picker still
opens on it, where rows arriving in disk order is the point (the listing is
usable before enumeration finishes).

Out of scope and left alone: the titles ("Choose File Notes Folder" is Title
Case and "File Notes" is internal jargon) belong to the copy branch, and the
"Loaded · 17 entries" miscount is filed with the import nits.

**RED->GREEN.** `Tests/UI/test_library_notes_w5_picker_followon.py`, 6 tests
for this task: all 6 red at `origin/dev` 48d40df8ce
(`AssertionError: "Keep a folder synced" must push the folder-only picker; it
pushed FileOpen`; `three doors, 2 different hints: {'Enter Open  ·  Select to
use this folder', 'Enter Open  ·  Select folder to use this folder'}`;
`KeyError: 'folders'`; `assert ['Reading', ..., 'README.md'] == ['Archive',
..., 'scratch.txt']`), 6/6 green here. The negative controls -- a file-only
`FileOpen` and a `FileSave` keeping `"discovery"`, and a file-only `FileOpen`
with an empty hint -- pass on BOTH sides, which is what makes them controls.

**Both-sides comparison.** `Tests/Library/` in full: identical FAILED name
sets. The 23-file picker test set, run sequentially: branch 1 failed / 510
passed against dev 19 failed / 492 passed, the branch's single red shared with
dev. One test in `Tests/UI/test_library_notes_wave_import_ux.py` asserted this
door pushes a `FileOpen`; it now asserts `SelectDirectory`, with the reason on
the line.

**Files.** `Third_Party/textual_fspicker/base_dialog.py`,
`select_directory.py`, `file_open.py`,
`parts/progressive_directory_navigation.py`, `Utils/input_validation.py`,
`UI/Library_Modules/library_notes_controller.py`,
`Tests/UI/test_library_notes_w5_picker_followon.py` (new). Guides and
`ENHANCEMENTS.md` §9 land with task-32643 in the same PR, since the two tasks
rewrite the same paragraphs.
<!-- SECTION:NOTES:END -->
