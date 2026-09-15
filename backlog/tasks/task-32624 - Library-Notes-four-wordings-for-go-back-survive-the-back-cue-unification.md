---
id: TASK-32624
title: 'Library Notes: four wordings for go back survive the back-cue unification'
status: Done
assignee:
  - '@robert'
created_date: '2026-09-15 06:44'
updated_date: '2026-09-15 18:50'
labels:
  - library
  - notes
  - critique-4
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 (dev 77eb2601a6), assessor A heuristic 4, every persona. Residual of task-32553 (PR #2685), which unified the back cue to one spelling.

What happened. A counted four back-control wordings still reachable inside the sub-screen: '‹ Notes', '‹ Back to list', 'Back', and Escape-only surfaces with no rendered control at all (A caps 17, 26 and the editor and sync panes). B sees the unified '‹ Notes' and '‹ Files' where the wave touched them (B caps 11, 48), so the unification landed where it was applied and stopped there.

The Escape ladder itself is correct and undocumented: Escape from the notes list focuses the rail, from the editor returns to the list, from Info returns to the editor -- three destinations, one key, with nothing on screen teaching the ladder (A section 11).

Cause PROVEN by capture.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 One spelling of the back control across every pane of the sub-screen
- [x] #2 Every pane that Escape leaves renders a back control, or the footer names where Escape goes from here
- [x] #3 The pins from task-32553 are extended to the panes it did not cover
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Grep library_notes_add_from_files_canvas.py (the "Add files"/lasting-sync canvas -- the chooser + setup + review + receipt + history panes) for every "notes-sync-back"/back-ish button and its exact label source.
2. Fix the two bare "Back" instances (configure and review phases) to back_cue_label("Notes"), matching the choose and receipt phases already on this same canvas.
3. Trace the Escape ladder for this canvas (LibraryNotesController._exit_library_notes_lasting_sync + LibraryScreen._library_notes_footer_shortcuts's lasting_add/lasting_roots branch) and the editor's own ladder (LIBRARY_NOTES_EDITOR_SHORTCUTS) to verify every phase either renders a back-cue button or the footer names "esc back to notes"/"esc notes" -- or, for the two phases where Escape is genuinely locked (checking/activating), that the footer honestly says so instead of claiming a dead key.
4. Extend task-32553's existing pin (test_one_back_cue_grammar_across_the_notes_surfaces, which only ever composed the chooser's default "choose" phase) to the "configure" and "review" phases it never reached.
5. Prove the new pin red against a reverted copy of the fix, then restore.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause of AC#1: LibraryNotesAddFromFilesCanvas._compose_pinned_actions
composes the SAME "#notes-sync-back" button on four phases -- "choose" and
"receipt" already called back_cue_label("Notes") ("‹ Notes"), but
"configure" (the Keep-a-folder-synced setup pane) and "review" still had a
bare "Back" left over, the one spelling task-32553's unification (which
touched three OTHER surfaces: the note editor, the import stepper, Session
Git) never reached inside this canvas. Fixed both to back_cue_label("Notes"),
matching their two siblings on the same canvas.

AC#2 (verified, not code-changed): traced the Escape ladder for both
surfaces this task names.
- Lasting-sync canvas (LibraryNotesController._exit_library_notes_lasting_sync):
  every phase either renders the back-cue button (now uniformly "‹ Notes"
  after the AC#1 fix) or is "checking"/"activating", where Escape is a
  genuine no-op (the method returns False without leaving) and the footer
  (LibraryScreen._library_notes_footer_shortcuts's lasting_add/lasting_roots
  branch) honestly shows "wait current step" instead of claiming a dead
  "esc" key -- the same honesty-lock pattern the delete-confirmation and
  conflict-resolution states already use elsewhere on this screen. The
  "history" sub-phase's own "Return" button goes to the enclosing review (a
  narrower action than Escape's "back to notes"), but the footer's generic
  "esc back to notes" for that phase is still true -- Escape really does
  exit all the way out, not to the review.
- Note editor ("editor" region): LIBRARY_NOTES_EDITOR_SHORTCUTS always
  carries ("esc", "back to notes") regardless of which control inside the
  region has focus (task-32623's ctrl+end fix only narrowed that OTHER
  entry, not this one), so the footer names Escape's destination on every
  width even where #library-note-back itself is conditionally hidden
  (#library-notes-task-return substitutes on wide terminals with a focused
  task, per task-19602).
No further code change found necessary for this AC.

AC#3: extended Tests/UI/test_library_notes_w4_import_keyboard.py's
existing task-32553 pin (test_one_back_cue_grammar_across_the_notes_
surfaces), which only ever composed the chooser canvas's default "choose"
phase, with a new parametrized test covering the "configure" and "review"
phases specifically -- the two this task found still wrong. Proved red:
reverted the back_cue_label("Notes") calls to the old bare "Back" in a
scratch backup-and-restore (backup taken first, restored immediately
after), reran, saw both parametrize cases fail with "Back" != "‹ Notes",
restored the fix.

Full suite: Tests/UI/test_library_notes_w4_import_keyboard.py (the file
housing both the task-32553 and the new task-32624 pins) = 3 relevant
tests passed, 0 failed.

Modified: tldw_chatbook/Widgets/Library/library_notes_add_from_files_canvas.py,
Tests/UI/test_library_notes_w4_import_keyboard.py.
<!-- SECTION:NOTES:END -->
