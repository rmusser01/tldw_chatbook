---
id: TASK-32552
title: >-
  Library Notes: Folder files polish — Ctrl+End inert in its editor, .trash
  listed but .obsidian hidden, embeds wrap at 100x30, Escape drops the whole
  mode in one press
status: To Do
assignee: []
created_date: '2026-09-13 06:47'
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
- [ ] #1 Ctrl+End and Ctrl+Home work in the Folder files editor and its footer advertises them, as the Library editor's does
- [ ] #2 The hidden-folder rule is stated in file-notes.md and applied consistently (.obsidian and .trash both hidden or both shown)
- [ ] #3 Escape from the Folder files editor first returns to the tree; a second Escape leaves the mode, and the footer says which
- [ ] #4 Embed lines do not wrap mid-token at 100x30
<!-- AC:END -->
