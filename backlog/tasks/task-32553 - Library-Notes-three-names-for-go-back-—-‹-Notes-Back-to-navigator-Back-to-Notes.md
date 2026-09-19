---
id: TASK-32553
title: >-
  Library Notes: three names for "go back" — ‹ Notes, Back to navigator, Back to
  Notes
status: Done
assignee: []
created_date: '2026-09-13 06:48'
updated_date: '2026-09-14 19:50'
labels:
  - library
  - notes
  - critique-3
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 (dev 5fd502dbac), assessor A, everyone. Task-32139 aligned ‹ Notes / ‹ Note inside the editor; the other two surfaces still use their own words.

**What happened.** `‹ Notes` (editor, A 05), `Back to navigator` (Session Git panel, A 83), `Back to Notes` (Add from files chooser and import stepper, A 25; B 24). Captures: A 05, 25, 83; B 24.

**Cause.** PROVEN copy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 One back-cue grammar is used by the editor, the Session Git panel and the import stepper
- [x] #2 notes.md and file-notes.md name the cue and are stamped
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce live (done: import stepper reads 'Back to Notes'; Session Git panel and Folder-files work pane read 'Back to navigator').
2. RED pin for one back-cue grammar across the editor, the import stepper, Add from files, the sync-roots list, the Session Git panel and the Folder-files work pane.
3. Fix: one back_cue_label(destination) helper in library_shell_state.py (no widget-to-widget import cycle); _library_note_back_label routes through it so the editor keeps its compact wording; the six sites become '‹ Notes' / '‹ Files'.
4. GREEN + live capture of the three cues + notes.md / file-notes.md stamps.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
One helper, `back_cue_label(destination)` in `Library/library_shell_state.py`, returns '‹ ' plus where the control goes. It lives in the shell-state copy module beside `library_disabled_action_label` rather than in `library_notes_canvas.py` as the brief sketched: the Session Git panel and the Folder-files workspace both already import that module, while importing the notes canvas from them would add a widget-to-widget dependency (and `library_notes_canvas` imports the import canvas, so the arrow only runs one way).

Six call sites route through it: the import stepper, the Add-from-files chooser (both phases), the lasting-roots list and its 'Nearest valid action' sentence become '‹ Notes'; the Session Git panel and the Folder-files work pane become '‹ Files'. The editor's own `_library_note_back_label` also routes through it, so task-32139's compact/wide split is unchanged ('‹ Notes' wide, '‹ Back to list' compact) and now shares one glyph with everything else.

**Consequence worth knowing.** The Session Git header's back cue lost 10 cells, which moved that panel's `-stack-actions` threshold from between 40 and 70 columns to between 32 and 34. `test_action_controls_fit_from_visible_label_cells_and_recompute` re-picks the width (and now pins BOTH sides of the threshold, where it crossed it once); its fit assertion moves to a width the panel can actually serve, since `#file-notes-git-bulk-toggle` is 35 cells wide whatever the stack class does. Fit at the narrowest supported width stays pinned by `test_focused_controls_keep_complete_labels_and_fit[(40, 20)]`.

**Files.** `Library/library_shell_state.py`, `Widgets/Library/library_notes_canvas.py`, `library_notes_add_from_files_canvas.py`, `library_notes_sync_roots_canvas.py`, `library_file_notes_git_panel.py`, `library_file_notes_workspace.py`, `Tests/UI/test_library_notes_w4_import_keyboard.py`, `Tests/UI/test_library_file_notes_workspace.py`, `Tests/UI/test_library_file_notes_git.py`, `Tests/Widgets/Library/test_library_notes_canvas.py`, `Docs/User_Guide/library/notes.md`, `Docs/User_Guide/library/file-notes.md`.
<!-- SECTION:NOTES:END -->
