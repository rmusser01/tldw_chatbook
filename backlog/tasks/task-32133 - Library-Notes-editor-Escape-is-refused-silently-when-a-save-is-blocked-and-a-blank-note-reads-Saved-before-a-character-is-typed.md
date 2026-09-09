---
id: TASK-32133
title: >-
  Library Notes editor: Escape is refused silently when a save is blocked, and a
  blank note reads Saved before a character is typed
status: Done
assignee: []
created_date: '2026-09-08 21:39'
updated_date: '2026-09-09 15:47'
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

Fix round 1 (Important 1 + 2): the notify was only wired at the Escape caller -- three OTHER callers of the shared `_exit_library_note_editor_guarded` seam (the "‹ Notes"/"‹ Back to list" button's handle_library_note_back, action_library_note_editor_back, and the wide task-return control's _return_from_library_notes_task) discarded the same False silently. Moved the notify INTO `_exit_library_note_editor_guarded` itself (library_screen.py) so every caller gets it; simplified action_library_notes_escape back to a bare call. Also: the mandated "fix the title or press Discard new note" copy was being applied to every NoteFlushOutcomeKind (VALIDATION_VETO, FAILED, CONFLICTED, BLOCKED, STALE) -- added a new `_library_note_editor_exit_veto_message(kind)` module function (library_screen.py) giving each kind its own real-state-plus-next-step sentence; only VALIDATION_VETO keeps the original mandated text. New test presses the "‹ Back to list" BUTTON (not Escape) with `_flush_library_note_save` mocked to return a CONFLICTED outcome, pinning both the shared-seam reach and the non-validation copy. Files: tldw_chatbook/UI/Screens/library_screen.py, tldw_chatbook/UI/Library_Modules/library_notes_controller.py.

PR #2547 review (Qodo finding 5): handle_library_note_keywords_changed (the wide field) clears _library_note_pending_blank_gc_id before scheduling autosave; the sibling handle_library_note_context_keywords_changed (Info's properties field) did not -- a fresh blank note edited only through Info's keyword field could autosave while the status stayed frozen at 'Draft -- not saved yet' (_library_note_is_pending_blank never went false). Added the same clear to the Context handler. New test: test_keyword_only_edit_through_info_clears_the_draft_status.

PR #2547 review (Qodo finding 2): added a direct parameterized unit test, test_exit_veto_message_covers_every_non_permitted_outcome_kind, pinning _library_note_editor_exit_veto_message's copy for every non-PERMITTED NoteFlushOutcomeKind (VALIDATION_VETO and CONFLICTED were previously only exercised indirectly through the full-editor UI tests; FAILED, BLOCKED, and STALE had no assertion at all).
<!-- SECTION:NOTES:END -->
