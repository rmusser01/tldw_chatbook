---
id: TASK-32253
title: >-
  Library Notes editor: Shift+Tab into the Title selects the whole title, so
  one keystroke destroys it and autosave commits the loss
status: Done
assignee: []
created_date: '2026-09-10 18:05'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - keyboard
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reproduced twice: `Ideas for study decks` became `!` on a single keypress after Shift+Tab moved focus into the Title, and autosave committed the loss (`R/caps/08`, `09`).

This is Textual's `Input.select_on_focus` default and it matches browser behaviour, which is exactly why it is filed at P2 rather than higher. What makes it a defect and not a default is that the pairing is backwards: the one field on the screen whose content must not be destroyed selects on focus, while the path fields that would genuinely benefit from select-on-focus do not (task-32251). Filed together, they are one decision about the same widget default, not two.

Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-10T17-32-16Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev e6cb464239, 2026-09-10, dual-assessor live review at 235x52 and 100x30/60x24 on two fresh profiles and one seeded profile with a git-backed 71-file Obsidian vault fixture; the reconciling parent re-tested every disagreement on a third pair of profiles and traced each surviving cause to code). Captures under the session scratchpad `notes-crit2/{C,D,R}/caps/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Moving focus into the note Title by keyboard places the caret without selecting the existing title
- [x] #2 AMENDED (fix round 1, review F7 — originally "a title replaced in one keystroke is recoverable: either undo inside the field restores it, or the change is not autosaved until the field is left"): no single keystroke can replace the title in the first place. Both original branches were unavailable without new machinery this task does not ask for — Textual 8.2.8's `Input` has no undo (no `action_undo`, no `ctrl+z` binding), and deferring the autosave until the field is left would change the autosave contract for every note field. AC#1's fix removes the loss this criterion guards against rather than adding recovery after it.
- [x] #3 Covered by a test: Shift+Tab into the Title followed by one character leaves the title intact apart from that character
<!-- AC:END -->

## Implementation Plan
<!-- SECTION:PLAN:BEGIN -->
1. Reproduce live: Shift+Tab from the body into the Title, then one character.
2. RED test on the real editor route.
3. Turn select-on-focus off on the shared note-field widget and park the caret at the end.
4. GREEN; verify live; guide; stamp.
<!-- SECTION:PLAN:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
Cause PROVEN: Textual's `Input.select_on_focus` defaults to `True`, and `NoteEditorInput` took the default. Reproduced live at dev 4a14b3f36f — Shift+Tab from the body into the Title, one `!`, and "Ideas for study decks" became "!" with the status line reading "Saved 15:23" a moment later (`wave3-caps/editor-keys/33-shifttab-title.txt`, `34-title-wiped.txt`).

Fix: `NoteEditorInput` defaults `select_on_focus=False` and, when it is off, parks the caret at the end of the existing text on focus. Arriving by keyboard now extends the title instead of arming its destruction. The two keyword boxes share the class and the rule; the path fields that genuinely want select-on-focus are a different widget (task-32251, another group).

AC#2 ("a title replaced in one keystroke is recoverable") is satisfied by removing the premise rather than adding recovery: no single keystroke can replace the title any more. That is the honest reading — Textual's `Input` has NO undo in 8.2.8 (checked: no `action_undo`, no `ctrl+z` binding), so an undo-based reading of AC#2 was not available without building one, which the AC does not ask for.

Verified live at 235x52: Shift+Tab into the Title then `ZZ` gives "Long note 35kZZ".

Files: `tldw_chatbook/Widgets/Library/library_notes_canvas.py`, `Tests/UI/test_library_notes_wave_editor_keys.py`, `Docs/User_Guide/library/notes.md`.
<!-- SECTION:NOTES:END -->
