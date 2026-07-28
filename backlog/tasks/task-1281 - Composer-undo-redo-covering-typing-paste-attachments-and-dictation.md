---
id: TASK-1281
title: 'Composer undo/redo (ctrl+z / ctrl+shift+z) covering typing, paste, file segments and dictation uniformly'
status: To Do
assignee: []
created_date: '2026-07-28 15:00'
labels: [console, composer, ux]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Console composer has no undo/redo. Draft text is mutated through several independent code paths -- direct typing (`composer.insert_text` from `on_key`'s printable-character branch), pasted text (`on_paste`), file/attachment segment insertion, and dictation transcript insertion (`ChatScreen._insert_console_dictation`) -- and a user who dictates or pastes the wrong thing, or fat-fingers a paste, currently has no way to revert just that change short of manually editing the draft back.

`ChatScreen.on_key` already owns an ad hoc whitelist of edit keys (`ctrl+a`, `ctrl+c`, `backspace`/`ctrl+h`, `delete`, arrow/home/end, `ctrl+w`, `shift+enter`/`ctrl+j`, `enter`, `pageup`/`pagedown`, `ctrl+u`) handled conditionally on composer focus/capture state, rather than through Textual's screen-level `BINDINGS` -- `on_key` is what lets the same physical keys behave differently depending on whether the composer is the active input target. Undo/redo needs to integrate through that same routing, immediately after `ctrl+u` (clear draft), so it activates only when the composer owns the keystroke.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 `ctrl+z` and `ctrl+shift+z` are registered in `ChatScreen.on_key`'s existing key-whitelist (next to `ctrl+u`), not added to `BINDINGS`.
- [ ] #2 Undo reverses the most recent composer draft mutation, whether it came from typing, paste, a file/attachment segment insertion, or a dictation transcript insertion (`_insert_console_dictation`).
- [ ] #3 Redo restores a mutation that was just undone.
- [ ] #4 Undo/redo history is scoped per Console session/draft and does not leak across switching to a different session.
<!-- AC:END -->
