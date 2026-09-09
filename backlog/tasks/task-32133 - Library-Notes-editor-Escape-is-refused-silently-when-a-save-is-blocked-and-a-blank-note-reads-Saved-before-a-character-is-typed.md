---
id: TASK-32133
title: >-
  Library Notes editor: Escape is refused silently when a save is blocked, and a
  blank note reads Saved before a character is typed
status: Done
assignee: []
created_date: '2026-09-08 21:39'
updated_date: '2026-09-09 06:58'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - copy
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Evidence assessor: with a title failing validation ('Title begins or ends with whitespace…'), Escape does nothing and says nothing; 'Discard new note' had already disappeared so the note had no exit except fixing the title. Design assessor: a new Blank note shows status 'Saved' and a list row 'Untitled' before anything is typed, and is then discarded if abandoned. Both halves are dishonest in opposite directions. Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-09T04-24-50Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev c4a7b1911f, 2026-09-08/09, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile with a 71-file Obsidian-style vault; parent re-tested every disagreement on a third profile).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A refused Escape states why and what to do ('Fix the title to leave, or Discard')
- [x] #2 A fresh blank note shows a draft state until its first save actually lands
- [x] #3 Covered by tests
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Escape: _exit_library_note_editor_guarded already returned a bool (True=exited, False=veto) but action_library_notes_escape discarded it. Now checks the result and notifies with the exact required copy on a veto: "Can't leave yet -- fix the title or press Discard new note." (severity=warning). Blank-note status: found the EXISTING _library_note_pending_blank_gc_id flag (already used to drive title_placeholder_only) marks a note as the untouched blank-GC candidate -- reused it (new _library_note_is_pending_blank helper, also deduping the inline condition previously only used for title_placeholder_only) to override both _library_note_status_line() and the resolve_database_note_status_channels() content_recovery with 'Draft -- not saved yet' until the user types anything or saves explicitly (both already clear the pending-blank flag synchronously). Files: tldw_chatbook/UI/Library_Modules/library_notes_controller.py. Tests: Tests/UI/test_library_notes_wave_editor_keys.py (2 new tests). Live-verified: whitespace-padded title -> Escape -> toast with exact copy; fresh Blank note -> status 'Draft -- not saved yet' until typed.
<!-- SECTION:NOTES:END -->
