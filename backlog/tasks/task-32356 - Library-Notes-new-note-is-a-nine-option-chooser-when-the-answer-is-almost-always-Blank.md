---
id: TASK-32356
title: >-
  Library Notes: new note is a nine-option chooser when the answer is almost
  always Blank
status: Done
assignee: []
created_date: '2026-09-11 06:17'
updated_date: '2026-09-11 08:31'
labels:
  - library
  - notes
  - ux
  - critique-10
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
ctrl+n yields Blank note plus eight dated templates (A cap 10). Blank is focused and Enter takes it, but the chooser still costs a read of nine items at the moment of lowest patience. Evidence: critique #10 snapshot (.impeccable/critique/2026-09-11T06-13-11Z__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 1f3184655b, captures under the session scratchpad crit10/A and crit10/B).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A new note is created immediately, with templates offered inside the editor or behind one 'From a template…' row
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Failing test: ctrl+n lands in the editor with no chooser; the create canvas offers one 'From a template…' row that opens eight.
2. Canvas create mode: Blank note + one 'From a template…' opener (templates_open disclosure, same shape as Collections quick capture).
3. action_library_notes_new creates a blank note directly (shared with the Blank note button); the create canvas stays on the rail's Create > New note row.
4. Update every pin that reached the editor via the chooser.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`n` and `ctrl+n` (both bound to `library_notes_new` since task-32138) now create the blank note and open its editor instead of routing to the nine-row Create chooser. `action_library_notes_new` still selects the Notes rail row first, so every guard a route change owns (File Notes flush, source normalization, the Notes session's own exit flush) runs exactly as before; only the destination moved, from Create to the notes list the new editor belongs to. The Blank-note button's body was extracted into `_start_library_blank_note` so the key and the button are one create, not two.

The Create canvas keeps the templates and stays reachable from the rail's Create > New note row, the Notes list's New button and the landing hub's New note action. Its eight dated template rows now fold behind a single 'From a template…' row (`#library-note-from-template`); `templates_open` lives on the canvas widget itself (nothing outside it reads the state, so it needs no LibraryNotesState field or kwargs plumbing) and resets whenever the canvas leaves create mode. Pressing the opener calls `preserve_same_id_focus_after_recompose()` first -- without it the recompose dropped focus onto the Items pane grip and a keyboard user could not walk into the rows it had just revealed (measured, not inferred). The create-mode status line reads 'Next: Start typing, or choose a template.'

Deviation from the plan's sketch: the plan's first test asserted focus lands on `#library-note-body`. It does not -- `_create_library_note` focuses `#library-note-title`, pinned by test_library_shell.py ('Successful Create never focused the title field') and deliberate since LIB-14, whose placeholder-only title exists so the first keystroke names the note. The plan's prose never asked for a focus change, so the pin stands and the test asserts the title.

Modified: tldw_chatbook/Widgets/Library/library_notes_canvas.py, tldw_chatbook/UI/Library_Modules/library_notes_controller.py, Docs/User_Guide/library/notes.md. Tests: new Tests/UI/test_library_crit10_notes_details.py; re-pointed create-path pins in test_library_notes_wave_editor_keys.py, test_library_crit8_keyboard.py, test_library_notes_riders_r_editor.py, test_library_shell.py (the key routes, and the canvases that enumerate template rows).
<!-- SECTION:NOTES:END -->
