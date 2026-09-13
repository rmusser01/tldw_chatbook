---
id: TASK-32246
title: >-
  Library Notes editor: Tab out of the note body lands on a control that
  swallows typed keys, and may leave the pane with no focus at all
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
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean repro on the seeded profile: open the 35 KB note from the list (the body is focused on open, heavy `+==+` border), press `Tab`, `Ctrl+End`, type `TAILEDIT`. DB content length unchanged at 35,182, `instr(content,'TAILEDIT') = 0`, no error, footer unchanged (`R/caps/13`, `14`). The capture taken after `Tab` shows the body reverted to the unfocused treatment and **no focused control anywhere in the visible pane**. D reproduced the same (D cap 46). Tab is the natural key to press here and the loss is total and unannounced; the guide's claim that "nothing repaints the editor underneath you... keystrokes never land in the wrong box" is contradicted.

Scope split, so nothing is fixed twice. The **burst** half of this mechanism is already covered by peer task-32106 / PR #2571: `_NOTE_FIELD_TAB_BINDINGS` in `library_notes_canvas.py`, carried by `NoteEditorInput` (title and both keyword boxes) and `NoteEditorTextArea` (the body), pinned by `test_a_body_tab_burst_leaves_the_trailing_word_out_of_the_body` and mutation-checked. That PR deliberately leaves the screen-level `tab` -> `focus_next` binding alone, because making it `priority` would pre-empt the delete-prompt Tab trap (`library_screen.py:7943`, pinned by the "Tab is trapped" group in `test_library_notes_wave_editor_keys.py`). Cite it; do not redo it.

What remains, and is this task:

1. A focus-**order** defect. After #2571, Tab from the body lands synchronously on a Button (`#library-notes-source-database` in the fixture), which silently swallows typed characters at any typing speed. The keystrokes are not racing a binding any more -- they are being delivered to a control that has nothing to do with them and says nothing about it.
2. The reconciler's observation that after Tab there was **no focused control anywhere on the pane** may be a distinct third thing from both the burst and the button landing. It is reproduced twice but not traced, and it must be either reproduced with an exact key sequence on record or ruled out.

Cause INFERRED for both remaining halves.

Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-10T17-32-16Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev e6cb464239, 2026-09-10, dual-assessor live review at 235x52 and 100x30/60x24 on two fresh profiles and one seeded profile with a git-backed 71-file Obsidian vault fixture; the reconciling parent re-tested every disagreement on a third pair of profiles and traced each surviving cause to code). Captures under the session scratchpad `notes-crit2/{C,D,R}/caps/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Tab out of the note body lands on a control that either accepts text or shows focus by shape
- [x] #2 Typing after a Tab out of the body never disappears silently: the characters either land somewhere visible or the footer names where focus is
- [x] #3 Covered by a test on the real editor route that fails on the live behaviour before the fix
- [x] #4 The no-focused-control-anywhere case is reproduced or ruled out, with the exact key sequence recorded in the notes
<!-- AC:END -->

## Implementation Plan
<!-- SECTION:PLAN:BEGIN -->
1. Reproduce live on a seeded profile: open a 37 KB note, Tab out of the body, type.
2. Trace the landing control and decide whether "no focused control anywhere" is a third defect.
3. RED test on the real editor route; scope Tab to the open editor at the existing focus seam.
4. Name the landing control in the footer through the existing `_library_focus_enter_label` seam.
5. GREEN; verify live at 235x52 and 100x30; guide key table; stamp.
<!-- SECTION:PLAN:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
Cause PROVEN for the focus-order half and the AC#4 half RULED OUT, both live at dev 4a14b3f36f on a seeded profile (captures `wave3-caps/editor-keys/10-tab-from-body.txt`, `11-tab-then-type.txt`).

`#library-note-body` is the LAST focusable widget of `#screen-content`, so `LibraryScreen.action_focus_next` wrapped the cycle round to that region's FIRST — `#library-notes-source-database`, the browse chrome's "Library notes" source switch, two panes above the editor. The harness focus chain confirms the order. A `Button` swallows printable keys, so `TAILEDIT` typed straight after Tab vanished with no error.

AC#4's "no focused control anywhere in the visible pane" is NOT a third defect. Exact sequence: open the note from the list, press Tab. Focus IS on `#library-notes-source-database`; a colour decode of that row (`tmux capture-pane -e`) shows it painting `1;4` bold+underline on `48;2;16;49;75` — which is the same treatment it already wears for its own `-selected` class, so a reader (and the reconciler) sees no change. It is invisible, not absent.

Fix: Tab and Shift+Tab cycle inside `#library-note-work-pane` while the note editor is open, the way they already cycle inside the delete prompt (`LibraryScreen.on_key`) and inside `#screen-content` (task-32052). One Tab from the body now lands on `‹ Notes`, which paints a heavy focus border. F6 and Escape remain the ways out of the editor, and the guide says so.

AC#2 needs typed characters to land visibly or the footer to name focus. The landing control is a Button, so the footer names it: `_LIBRARY_NOTE_EDITOR_ENTER_LABELS` feeds `_library_focus_enter_label` (the seam the delete prompt and the New-note canvas already use) and the editor footer tier appends an `enter …` chip whenever a named control has focus. Appended, not prepended (fix round 1, review F1): the real footer keeps only the leading chips that fit, so the exit heads the tier and the focus chip is the one dropped when the width budget bites. Live: `esc back to notes | ctrl+end end of note | enter back to list`; at 60 columns `esc notes | ctrl+end end` (pinned by `test_the_editor_exit_chip_survives_at_sixty_columns`).

Files: `tldw_chatbook/UI/Screens/library_screen.py`, `Tests/UI/test_library_notes_wave_editor_keys.py`, `Docs/User_Guide/library/notes.md`.
<!-- SECTION:NOTES:END -->
