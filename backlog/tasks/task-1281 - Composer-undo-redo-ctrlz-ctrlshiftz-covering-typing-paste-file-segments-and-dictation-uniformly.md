---
id: TASK-1281
title: >-
  Composer undo/redo (ctrl+z / ctrl+shift+z) covering typing, paste, file
  segments and dictation uniformly
status: Done
assignee:
  - '@claude'
created_date: '2026-07-28 15:00'
updated_date: '2026-07-30 19:30'
labels:
  - console
  - composer
  - ux
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
- [x] #1 `ctrl+z` and `ctrl+shift+z` are registered in `ChatScreen.on_key`'s existing key-whitelist (next to `ctrl+u`), not added to `BINDINGS`.
- [x] #2 Undo reverses the most recent composer draft mutation, whether it came from typing, paste, a file/attachment segment insertion, or a dictation transcript insertion (`_insert_console_dictation`).
- [x] #3 Redo restores a mutation that was just undone.
- [x] #4 Undo/redo history is scoped per Console session/draft and does not leak across switching to a different session.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Map every composer mutation method (insert_text, insert_text_as_paste, backspace/delete/ctrl+w handlers, clear_draft, attachment segment insertion) and ChatScreen's session-switch draft swap sites (~:5658-5687, :11859, :12289)
2. Snapshot-based undo/redo history inside ConsoleComposerBar: (text, cursor) pairs, consecutive single-char typing coalesced, any other mutation kind or cursor move breaks the run, depth cap 100; ctrl+u clear IS an undoable mutation; load_draft/clear_draft during session switch are scope changes, NOT history entries
3. export_undo_history()/restore_undo_history() on the composer; ChatScreen stores per-session histories alongside its existing set_session_draft dance
4. ctrl+z / ctrl+shift+z in ChatScreen.on_key whitelist next to ctrl+u (AC1); undo/redo re-persists the resulting draft to the console chat store (mirror _insert_console_dictation's :5192)
5. RED-first tests: undo of typing run / paste / dictation insertion / attachment segment / ctrl+u; redo; cross-session isolation; store consistency after undo; mutation-checked
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Shipped on feat/console-voice-control-v2 (commits cc3c1a4a2..eae02b7ed, 11 commits over 3 review
waves). History lives inside ConsoleComposerBar as flat (text, cursor) snapshots: typing coalesced,
per-stack char budget (~2M) plus 100-entry cap, ctrl+u undoable, post-ACCEPTED-send clears both
stacks (refused sends keep undo), session scoping via export/restore banked per session id on
ChatScreen with the same visible-draft guard as dictation insertion (undo is a full no-op in the
switch settle window). Background (store-only) dictation appends DROP the banked history rather than
risk destructive stale undo. Undo/redo re-collapses restored segments over a dedicated
UNDO_RECOLLAPSE_CHAR_THRESHOLD = 20,000 (perf guard, independent of the cosmetic
collapse_large_pastes pref — a 200KB restore froze the UI 2.9s before this), cursor snapped off
token interiors. Documented limitations: collapsed-token labels/display state are not carried
through flat snapshots (50-20,000-char pastes restore literal, repaint <=36ms); non-Kitty terminals
collapse ctrl+shift+z to ctrl+z at the wire level, so redo needs TASK-1500's ctrl+y alias.
Review trail: .superpowers/sdd/2026-07-29-console-voice-control-v2/task-1281-{report,review}.md.
41 tests in Tests/UI/test_console_composer_undo.py; dictation contract file byte-identical.
<!-- SECTION:NOTES:END -->
