---
id: TASK-32611
title: >-
  Library Notes: three folder pickers, one labelled File name while it can only
  pick a folder
status: To Do
assignee: []
created_date: '2026-09-15 06:39'
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
- [ ] #1 One picker component serves all three doors, with a mode flag for files-plus-folder versus folder-only
- [ ] #2 Folder-only mode labels its field for a folder and offers only the buttons that can act on one
- [ ] #3 All three doors arrive with the same focus, the same hint wording and the same button grammar
- [ ] #4 The default listing order is folders-first, name-ascending, with Discovery order kept as an option
<!-- AC:END -->
