---
id: TASK-32606
title: >-
  Library Notes: the Choose File Notes Folder dialog opens with no keyboard
  focus, so Folder files is mouse-only
status: To Do
assignee: []
created_date: '2026-09-15 06:37'
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
- [ ] #1 The Choose File Notes Folder dialog focuses its path field on mount and selects any pre-filled value, so the first keystroke lands in the field
- [ ] #2 The dialog renders its own footer chips, so a keyboard user can see which controls are inside it
- [ ] #3 Folder files can be opened, a folder chosen and a file edited with no mouse at all
- [ ] #4 The focus-on-mount behaviour is shared by every folder-offering dialog rather than overridden per subclass, and a test covers the Folder-files door specifically
<!-- AC:END -->
