---
id: TASK-32606
title: >-
  Library Notes: the Choose File Notes Folder dialog opens with no keyboard
  focus, so Folder files is mouse-only
status: In Progress
assignee: []
created_date: '2026-09-15 06:37'
updated_date: '2026-09-15 15:41'
labels:
  - library
  - notes
  - critique-4
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 (dev 77eb2601a6), assessor A P1, personas Sam (keyboard-only, low vision) and Jordan, Obsidian workflow. A hard blocker for Sam.

What happened. Folder files -> choose a folder opens 'Choose File Notes Folder' with focus outside the dialog. Reproduced twice from a clean open: typing a path immediately landed nothing, and three Tab presses left the footer on the Library chrome ('/ focus search | F6 next pane | esc notes') rather than any dialog control. Only a mouse click into the Folder path field made the dialog usable (A caps 27, 28). By contrast the Import once picker arrives with its field focused and typing works at once (A cap 18, B K13).

Cause, PROVEN for the focus half. Wave 4's task-32540 fix (PR #2685) added a _focus_initial_widget override to FileOpen, gated on offer_select_folder (Third_Party/textual_fspicker/file_open.py:74-97), which is what Import once and Keep a folder synced push. The Folder-files door pushes a different class -- SelectDirectory (Widgets/Library/library_file_notes_workspace.py:6468) -- and only FileSave and FileOpen override _focus_initial_widget (grep across Third_Party/textual_fspicker): the sibling caller the fix did not reach. That Tab does not enter the dialog at all is INFERRED (not traced to the modal's focus chain).

Docs contradicted: notes.md says the picker 'opens with that File name field already focused'. True for two of the three doors.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Choose File Notes Folder dialog focuses its path field on mount and selects any pre-filled value, so the first keystroke lands in the field
- [x] #2 The dialog renders its own footer chips, so a keyboard user can see which controls are inside it
- [x] #3 Folder files can be opened, a folder chosen and a file edited with no mouse at all
- [x] #4 The focus-on-mount behaviour is shared by every folder-offering dialog rather than overridden per subclass, and a test covers the Folder-files door specifically
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Probe the family headless: confirm SelectDirectory focuses the listing while FileOpen(offer_select_folder) focuses its field; enumerate every FileSystemPickerScreen subclass and which has the behaviour.
2. RED pins in Tests/UI/test_library_notes_w5_picker_keyboard.py on the real Folder-files route (chooser button -> SelectDirectory) plus the family-wide parametrised pin.
3. Hoist task-32540's FileOpen._focus_initial_widget into FileSystemPickerScreen behind one shared declarative flag; delete the FileOpen override; SelectDirectory declares the flag.
4. AC#2: the picker renders its own Footer so the modal's chips replace the Library chrome showing through the translucent ModalScreen.
5. GREEN + keyboard-only live walk at 235x52 and 100x30; file-notes.md keyboard route + stamp.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Folder files' "Choose File Notes Folder" now opens on its path field, and every
folder-returning picker in the family does the same through one shared seam.

**AC#1/#4 — the fix is where all callers route through.** Wave 4's task-32540
put a `_focus_initial_widget` override on `FileOpen`, gated on
`offer_select_folder`. Folder files pushes `SelectDirectory`, which never got
it — and neither did `EnhancedSelectDirectory` (Personas, vLLM model dir), the
second silent instance nobody had noticed. Rather than write a third override,
the behaviour moved onto `FileSystemPickerScreen._focus_initial_widget` behind
one declarative fact, `RETURNS_A_FOLDER`: the two directory-only dialogs
declare it as a class attribute, `FileOpen` answers it per instance through its
existing `_offer_select_folder`, and `FileOpen._focus_initial_widget` is
deleted. The field is read generically as the input bar's one `Input`, exactly
the way `_resolve_select_folder_target` already reads it, so the differing ids
(`#path_input`, `#dir-path-input`, the anonymous `FileNameInput`) need no
special-casing. `FileSave` keeps its own override: it is not folder-returning,
and its reason (confirm a seeded filename with one Enter, task-1479) is
different. File-only pickers are unchanged, pinned by negative controls on both
`FileOpen` and `EnhancedFileOpen`.

**RED→GREEN.** `Tests/UI/test_library_notes_w5_picker_keyboard.py`, 9 tests:
RED 6 failed / 3 passed at `origin/dev` 4e4558bff2 (the 3 that passed were
`FileOpen(offer_select_folder)` — already fixed by wave 4 — and the two
negative controls); GREEN 9/9. The Folder-files door is driven end to end: the
shipped `#file-notes-choose-root` button is focused and activated with
`pilot.press("enter")` inside the real `LibraryScreen`, and the assertion reads
what the real `SelectDirectory` focuses.

**AC#2 — two footers would have been worse than one.** A `ModalScreen` is
translucent, so the host screen's footer shows straight through it; critique #4
Tab'd three times inside the dialog and kept reading Library's
"/ focus search | F6 next pane | esc notes". The dialog now yields a `Footer`.
It is docked at SCREEN level, not inside `Dialog`: mounted inside the dialog it
merely added a second key row eight lines above a contradicting one (captured,
`02-picker-open-235x52.txt`), which is the "two instructions at once" defect
the same critique flags elsewhere. At screen level the opaque row REPLACES the
host's chips for as long as the modal is up. `Footer` lays chips out in binding
order and scrolls the overflow off the right edge, so `BINDINGS` was reordered
to lead with escape and the two path actions — before the reorder `esc Cancel`
fell off the edge entirely at 100 columns (`08` vs `09`).

**AC#3 — live, keyboard only, both sizes.** Scratch power profile + 65-file
vault, `TLDW_CONFIG_PATH` resolved and its data dir confirmed under the scratch
profile before each launch. At 235x52 and 100x30 the mode strip, Choose folder…,
the picker, the typed path and Select were all reached with
Tab/Shift+Tab/Enter and no click; a pasted absolute path REPLACED the pre-filled
value with no click first (`04`, `10`). Log grep after both walks: zero
`unhandled_exception`, zero `| ERROR`, two `app_stopping` (my own Ctrl+Q).
The remaining leg — opening a file from the Files tree — is pinned headless on
the real route (`test_folder_files_reaches_and_edits_a_file_with_no_mouse`)
rather than live: Tab from the folder navigator's search field leaks into the
Library rail, which is a separate pre-existing gap filed as a rider.

**Trade-off / known residue.** `^s Select this folder` now renders in the
dialog's footer, dimmed, on pickers that do not offer it — Textual's
`Footer` composes every `show=True` binding and marks a `check_action`-vetoed
one disabled rather than dropping it, which the task-2222 guard's comment
assumed it did. Making ctrl+s live on `SelectDirectory` would bypass
`EnhancedSelectDirectory`'s own select handler and its remembered-directory
bookkeeping, so it is left dimmed and documented; rider filed.

**Follow-on test repair, not a loosening.** Two task-32251 pins drove
"click an UNFOCUSED field" on `SelectDirectory`, which now arrives focused.
They focus the listing first to reach the click-to-focus seam — the identical
treatment task-32540 already applied to the `FileOpen` pin sitting beside them.
The seam under test is unchanged.

**Files.** `Third_Party/textual_fspicker/base_dialog.py`, `select_directory.py`,
`file_open.py`, `Widgets/enhanced_file_picker.py`,
`Tests/UI/test_library_notes_w5_picker_keyboard.py` (new),
`Tests/UI/test_picker_path_field.py`, `Docs/User_Guide/library/file-notes.md`.
No CSS changed (Textual's `Footer` plus the existing app-tier
`core/_base.tcss` rule); bundle sync, boot-CSS byte budget and preflight green.
<!-- SECTION:NOTES:END -->
