---
id: TASK-32247
title: >-
  Library Notes editor: Ctrl+End does not reach the end of a long note, in any
  encoding
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
Repro on the seeded profile with a 35 KB note: click into the body -- which does place the caret, since a marker typed after a click landed exactly where clicked -- then press `Ctrl+End` and type. The text landed at character **23** of 35,187 (`R/caps/15`). Retried with the raw xterm sequence `\x1b[1;5F`: character 28. D measured character 125 by the same route. Plain `End` is delivered, so this is not a general key-delivery failure.

The consequence is that the power user's stated task -- edit near the end of a long note -- cannot be done by keyboard at all, and no on-screen affordance offers an alternative.

Cause INFERRED: Textual's `TextArea` binds `ctrl+end` to `cursor_document_end`, so something upstream is swallowing it; not traced.

Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-10T17-32-16Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev e6cb464239, 2026-09-10, dual-assessor live review at 235x52 and 100x30/60x24 on two fresh profiles and one seeded profile with a git-backed 71-file Obsidian vault fixture; the reconciling parent re-tested every disagreement on a third pair of profiles and traced each surviving cause to code). Captures under the session scratchpad `notes-crit2/{C,D,R}/caps/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A document-end key moves the caret to the end of the note body from any starting position, verified on a note of about 35 KB
- [x] #2 The key is advertised in the editor footer beside the other editor keys
- [x] #3 Covered by a test that presses it on a multi-thousand-line body and asserts the resulting caret location
<!-- AC:END -->

## Implementation Plan
<!-- SECTION:PLAN:BEGIN -->
1. Reproduce live with all three Ctrl+End encodings on a 37 KB note.
2. Check Textual's own TextArea bindings before assuming something upstream swallows the key.
3. RED test on a multi-thousand-line body asserting the caret location.
4. Bind the key (and its Home twin) on `NoteEditorTextArea`; advertise it in the editor footer.
5. GREEN; verify live at 235x52 and 100x30; guide key table; stamp.
<!-- SECTION:PLAN:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
Cause PROVEN, and it is NOT the inferred one. The task inferred that Textual's `TextArea` binds `ctrl+end` to `cursor_document_end` and something upstream swallowed it. Textual 8.2.8 binds no such key and defines no such action: its only `end` binding is `"end,ctrl+e" -> cursor_line_end`, and `action_cursor_document_end` does not exist. There was never a key to swallow. Verified twice — in the widget source, and in a harness probe where `"ctrl+end" in body._bindings.key_to_bindings` is `False` and the caret does not move.

Reproduced live at dev 4a14b3f36f on a 37,099-character note: `\x1b[1;5F` then `TAILEDIT` left the text at character 0 (`wave3-caps/editor-keys/08-ctrlend-live.txt`). All three encodings the critique tried resolve to the single Textual key name `ctrl+end` (`ANSI_SEQUENCES_KEYS["\x1b[1;5F"] == (Keys.ControlEnd,)`), so one binding covers them; that mapping is pinned so a Textual upgrade that renames it fails loudly.

Fix: `NoteEditorTextArea` carries `ctrl+end`/`ctrl+home` and the two actions they need. AC#2: the editor footer tier now reads `esc back to notes | ctrl+end end of note` (compact: `esc notes | ctrl+end end`). Escape stays FIRST in both tiers — unlike its sibling tiers — because the real footer drops trailing chips that miss the width budget and the exit is the one that must survive. Fix round 1 (review F2) corrected this paragraph: round 0 cited `test_compact_editor_context_reaches_the_paint_at_60_cols`, which does not exist, and claimed a budget of one chip that is not this tier's. Measured through the real `AppFooterStatus` at 60 columns: BOTH compact chips paint (`esc notes | ctrl+end end`), a third is dropped, and the same pair in the WIDE wording paints only `esc back to notes` — so the head of the tier is what is guaranteed whatever a later label growth does. The real pins are `test_only_one_context_chip_paints_at_sixty_columns` (the budget) and the new `test_the_editor_exit_chip_survives_at_sixty_columns` (this tier against the task-32246 focus chip). The honesty contract requires both tiers to carry the same keys in the same order.

Verified live at 235x52 and 100x30: the raw `\x1b[1;5F` sequence lands the caret on line 00699 of 700 and the marker types there (`41-verify-ctrlend.txt`, `51-compact-ctrlend.txt`).

Files: `tldw_chatbook/Widgets/Library/library_notes_canvas.py`, `tldw_chatbook/UI/Screens/library_screen.py`, `Tests/UI/test_library_notes_wave_editor_keys.py`, `Tests/UI/test_library_shell.py`, `Docs/User_Guide/library/notes.md`.
<!-- SECTION:NOTES:END -->
