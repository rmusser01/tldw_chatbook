---
id: TASK-32552
title: >-
  Library Notes: Folder files polish — Ctrl+End inert in its editor, .trash
  listed but .obsidian hidden, embeds wrap at 100x30, Escape drops the whole
  mode in one press
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-13 06:47'
updated_date: '2026-09-13 15:13'
labels:
  - library
  - notes
  - critique-3
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 (dev 5fd502dbac), both assessors, personas Jordan and Riley, Folder files workflow. Grouped by surface (precedent: 32261 / 32262).

1. Ctrl+End does not move the caret in the Folder files editor: the typed text landed at the click — on disk `> Se` / `JORDAN edited this file via Folder filesnd the vault import…` (B D9, cap 50 + cat). Task-32247 bound ctrl+end on `NoteEditorTextArea` (the Library editor) only; file-notes.md does not list the key.
2. The tree lists `.trash` but hides `.obsidian` — one hidden folder shown, one not, no rule stated (A 49; B 40).
3. At 100x30 `![[attachments/diagram.png]]` wraps across lines mid-token (A 57).
4. Escape from the Folder files editor drops the whole Folder files mode in one press (A 58) — fast for Alex, surprising for Jordan.

**Cause.** INFERRED for all four.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Ctrl+End and Ctrl+Home work in the Folder files editor and its footer advertises them, as the Library editor's does
- [x] #2 The hidden-folder rule is stated in file-notes.md and applied consistently (.obsidian and .trash both hidden or both shown)
- [x] #3 Escape from the Folder files editor first returns to the tree; a second Escape leaves the mode, and the footer says which
- [x] #4 An embed line no longer than the editor pane does not wrap mid-token at 100x30 (controller ruling, wave 4: a token longer than the pane cannot fit any row, so the criterion is qualified to lines the pane can hold)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce live (fn-01..04): Ctrl+End inert, .trash listed / .obsidian hidden, embed wrap at 100x30, Escape leaves the mode in one press\n2. RED pins in Tests/UI/test_library_notes_w4_file_notes.py + a service test for dot-directories\n3. Fix: FileNotesEditorTextArea with ctrl+end/ctrl+home; hide every dot-directory in the service walk; workspace-level Escape that focuses the tree while the editor has focus; footer tier says esc files / esc notes and ctrl+end end of file\n4. Measure the editor pane at 100x30; qualify AC#4 to lines no longer than the pane\n5. GREEN, live captures, guide (keys table, hidden-folder rule, Escape ladder) + stamp
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Four defects on one surface, each fixed where all callers route through.

**Ctrl+End / Ctrl+Home.** The Folder files editor was a plain `TextArea`, and
Textual 8 binds `end`/`home` to the LINE ends and defines no `ctrl+end`
binding or `cursor_document_end` action at all -- the key was not swallowed,
it did not exist, so the text typed after it landed at the click.
`FileNotesEditorTextArea` adds the two bindings and the two actions. The four
lines are copied rather than `NoteEditorTextArea` imported: that class also
carries a priority Tab pair namespaced to the Library screen, which is not
this widget's business.

**Hidden folders.** The walk skipped only `.git`; `.obsidian` merely happened
to hold no supported file while `.trash/Old idea.md` is Markdown, so one
appeared and the other did not -- an accident, not a rule. The rule is now
explicit: every dot-directory is hidden. Folder files edits the folder in
place, so those directories stay exactly as they are on disk, just out of the
tree and out of search.

That narrowing had an upgrade path the fresh-fixture tests could not see. A
replica that indexed such a file BEFORE the rule reports it as missing on the
next walk, and reconcile tombstones anything missing: the first live run
listed `.trash/Old idea.md` under "Recently deleted" for a file still on disk.
Such a row is now FORGOTTEN (`FileNotesReplica.forget_file` -- row and FTS
entry dropped, no tombstone), and `_forget_hidden_tombstones` sweeps rows an
older build had already tombstoned, on both read seams, since reconcile only
revisits rows the walk still reported.

**Escape.** One press dropped the whole mode. A workspace-level
`Binding("escape", "focus_tree")`, gated by `check_action` to a focused
editor with a visible tree and no path task running, steps back to the files
tree; everywhere else the key falls through to the screen exactly as before,
so the second press still leaves.

**Footer.** `LIBRARY_NOTES_FILES_SHORTCUTS` lost its fixed Escape chip; the
Folder files branch splices in `esc files` / `esc notes` and `ctrl+end end of
file`. Both are editor-FOCUS facts, found live: an open-file gate left the
Ctrl+End chip standing after a folder change closed the file, and the footer
is only re-registered when `_refresh_footer_typing_context`'s context tuple
flips -- editor -> "File contents…" flips neither the typing flag (both text
widgets) nor the Enter label, so the stale `esc files` chip survived. The
tuple gains the editor-focus axis.

**Embeds (AC#4, qualified by the controller).** No code change, and the AC
text was amended first. Measured at 100x30 (`fn-40-embed-100x30.txt`): the
editor frame is 32 cells and its text wraps at 27, while
`![[attachments/diagram.png]]` is 28 cells of unbreakable token -- no row at
that width can hold it. Textual only splits a token that cannot fit a row by
itself, so what is true and worth pinning is the qualified property: every
line the pane CAN hold renders on one row. Widening the pane is
`LIBRARY_FILE_NOTES_READER_PROFILE.work_min_width` (30) against a files tree
holding ~59 columns -- a layout decision, not this task's.

RED proofs, each by patching the fix out of a scratch copy of the file (never
`git stash`, never a revert on the branch): plain `TextArea` -> caret stays at
(0,0); `name != ".git"` -> `.obsidian`/`.trash`/`.hidden-notes` back in the
scan; no workspace BINDINGS -> one Escape leaves the mode; the old constant
footer tuple -> `esc files`/`esc notes` both absent; no ctrl+end append -> no
chip; `current_path` instead of `editor_focused` -> chip still registered on
the search box; two-element context tuple -> registered footer keeps `esc
files`; no `_forget_hidden_tombstones` / no forget branch -> the file stays in
`list_deleted`.

Live (captures under `wave4-caps/file-notes/`): `fn-30`/`fn-40` (no
dot-folder in either tree), `fn-31-ctrl-end-235x52` + `fn-31-file-after`
(Ctrl+End then typing appends at EOF on disk; same at 100x30 via `fn-41`),
`fn-33`/`fn-34` and `fn-42`/`fn-43` (the two-step Escape and both footer
chips at 235x52 and 100x30), `fn-40-embed-100x30` (the wrap measurement). The
tombstone sweep was driven on a profile seeded with a `.trash/Old idea.md`
tombstone: after one scan `list_deleted` was empty and "Recently deleted"
named nothing.

Files: `tldw_chatbook/Widgets/Library/library_file_notes_workspace.py`,
`tldw_chatbook/Notes/file_notes_service.py`,
`tldw_chatbook/Notes/file_notes_replica.py`,
`tldw_chatbook/UI/Screens/library_screen.py`,
`Tests/UI/test_library_notes_w4_file_notes.py` (new pin file),
`Tests/Notes/test_file_notes_service.py`, `Tests/UI/test_library_crit9_notes.py`,
`Docs/User_Guide/library/file-notes.md`,
`backlog/docs/lessons-live-verification.md`.
<!-- SECTION:NOTES:END -->
