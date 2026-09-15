---
id: TASK-32608
title: >-
  Library Notes: Activate reviewed root and the Session Git commit buttons
  cannot be reached by keyboard
status: In Progress
assignee: []
created_date: '2026-09-15 06:38'
updated_date: '2026-09-15 17:59'
labels:
  - library
  - notes
  - critique-4
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 (dev 77eb2601a6), assessor B D2, P1, persona Sam (keyboard-only), Obsidian workflow. Two of B's six cells are not completable without a pointer.

What happened. (a) Sync review: Tab runs past 'Activate reviewed root' into the rail, and Enter there navigates the app to Conversations -- the root was never activated (B cap 33, K17). (b) Session Git commit form: the subject field and the Cancel commit / Review commit buttons are 30 rows apart, Tab x4 reaches no visible stop, and the buttons were only reachable by a computed mouse click at row 48 (B caps 42, 43, K18). A independently reports the same shape for the import review: no visible keyboard route to 'Import selected items' or 'Activate reviewed root', both reached with the mouse (A section 9).

Keystroke result: create, edit and Import once are fine (1, 4 and ~12 keys); lasting-sync activation and Session Git commit are not completable by keyboard.

Cause INFERRED -- screen-wide Tab order with no containment for these panes; not traced. No pinning test found. Adjacent open rider: 32585 (reaching the Manage sync folders controls takes about 23 Tabs), which names the same missing containment for the roots list.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every action in the sync review pane is reachable by Tab from that pane's first control, with a visible focus indicator, and Tab does not leak into the rail
- [x] #2 Every action in the Session Git commit form and commit review is reachable by Tab from the subject field
- [ ] #3 A keyboard-only walk completes activate-a-root and commit-a-session-change end to end, captured
- [x] #4 Tests pin the tab route into each pane's terminal action, so a later layout change cannot silently strand it
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace the Tab containment seam (_LIBRARY_WORK_PANE_TAB_VIEWS / _library_note_work_pane_owns_tab).
2. Add the two lasting-sync views so Tab closes inside the sync pane instead of leaking into the rail.
3. Trace the Session Git commit form's own Tab route and close the equivalent gap.
4. Headless keyboard walks (Tab-only) that reach 'Activate reviewed root' and the commit form's terminal actions.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Two cells, two different mechanisms, both closed at the seam that already owns the behaviour.

(a) Sync review. The lasting-sync canvas is mounted in the SAME `#library-note-work-pane` as the note editor (`LibraryNotesCanvas.compose` yields `LibraryNotesAddFromFilesCanvas` for the lasting_add / lasting_roots modes), but `_LIBRARY_WORK_PANE_TAB_VIEWS` listed only ('editor', 'import') -- so the closed Tab cycle task-32052/32540 built for the other full-pane tasks simply did not apply to it and Tab ran past 'Activate reviewed root' into the rail. Adding the two views is the whole fix; the gate keys on the VIEW, not the sync phase, so it covers every phase including review.

(b) Session Git commit. Measured before changing anything: from the subject field the form's own Cancel commit and Review commit are the second and third Tab stops -- they were never unreachable in the panel's own chain. What was unreachable was reaching them THROUGH the Library screen, whose Tab region is the whole of `#screen-content` and whose focus chain Textual orders by screen position, so the next stop from a field at the top of the pane is whatever sits below it anywhere on screen, and the form's footer 30 rows down is not it. Fixed with Textual's own `trap_focus` on the commit and push workflow containers, applied in `_sync_workflow_surfaces` -- the one place their display is already resolved, so both workflows get it and no second Tab mechanism is invented. Escape still leaves (binding resolution is not focus-scoped) and so does the form's own Cancel.

AC#3, scoped honestly: the commit half is done end to end from the keyboard in the pinned test -- type the subject, Tab to Review commit, Enter, and the draft leaves the form phase. The activate-a-root half is NOT a full end-to-end walk: headless coverage stops at 'the activate button is inside the pane's Tab cycle and Tab never leaves the pane'. The experiment that would settle it is the real-vault journey harness in Tests/UI/test_library_notes_files_sync_journey.py driven with pilot.press('tab')/'enter' instead of Button.press(); it was left out of this branch because two other wave-5 branches are editing the sync runtime and reconciler concurrently.

Files: UI/Screens/library_screen.py (`_LIBRARY_WORK_PANE_TAB_VIEWS`), Widgets/Library/library_file_notes_git_panel.py (`_sync_workflow_surfaces`), Tests/UI/test_library_notes_w5_kbd_focus.py, Docs/User_Guide/library/notes.md.
<!-- SECTION:NOTES:END -->
