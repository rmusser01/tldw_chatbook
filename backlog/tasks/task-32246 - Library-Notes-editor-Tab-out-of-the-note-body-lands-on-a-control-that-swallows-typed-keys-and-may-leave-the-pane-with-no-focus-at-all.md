---
id: TASK-32246
title: >-
  Library Notes editor: Tab out of the note body lands on a control that
  swallows typed keys, and may leave the pane with no focus at all
status: To Do
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
- [ ] #1 Tab out of the note body lands on a control that either accepts text or shows focus by shape
- [ ] #2 Typing after a Tab out of the body never disappears silently: the characters either land somewhere visible or the footer names where focus is
- [ ] #3 Covered by a test on the real editor route that fails on the live behaviour before the fix
- [ ] #4 The no-focused-control-anywhere case is reproduced or ruled out, with the exact key sequence recorded in the notes
<!-- AC:END -->
