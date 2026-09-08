---
id: TASK-32063
title: >-
  Library Notes/Import status prose: duplicated status line, run-on header, and
  a graduation notice painted as a bare line
status: Done
assignee: []
created_date: '2026-09-08 18:24'
updated_date: '2026-09-08 20:03'
labels:
  - library
  - notes
  - copy
  - ux
  - critique-8
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
'Library notes · Library database · Ready · Next: Create a note or add from files.' appears in the list pane and again as the canvas header; the Add-from-files header is a 130-character run-on plus a second header; 'Library tools are now available.' stays as an unframed line all session and also fires on a populated profile's first visit (any transition into GRADUATED, including UNKNOWN to GRADUATED on the first source read). Evidence: critique #8 snapshot `.impeccable/critique/2026-09-08T16-05-41Z__tldw-chatbook-ui-screens-library-screen-py.md` (dev 1c022378cb, 2026-09-08, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile). Register row 14.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each Notes canvas paints one status line, with 'Next:' only when it names a control on screen
- [x] #2 The graduation notice is a toast, fired only on a real compact-to-graduated transition
- [x] #3 The Add-from-files header is one sentence
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Failing tests: work pane restates the list pane's authority sentence; Add from files stacks four headers; the graduation notice fires on a populated profile's first visit.
2. Give LibraryNotesCanvas an overridable authority prefix; the work pane returns empty.
3. Drop the work pane's own authority line for modes whose child canvas paints one.
4. 'Next:' only when it names a control.
5. One-sentence Add-from-files header.
6. Gate the graduation notice on STARTER -> GRADUATED.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Three separate defects, all measured live at 235x52.

(1) LibraryNotesCanvas yields its authority line from compose(), and LibraryNoteWorkPane inherits it -- so a wide session painted 'Library notes . Library database . ...' twice, once per pane. The prefix is now an overridable _authority_prefix(); the work pane returns '' and paints no line at all for modes whose child canvas owns its header (import, lasting_add, lasting_roots). That last rule is what fixed Add from files, where four headers stacked over one idea (caps/05 before, caps/19 after). A 'Next:' clause now survives only where it names a control on screen -- 'Wait for loading/saving/export to finish' named none.

(2) The Add-from-files header is one sentence, 'Add files to Library notes.' (was a three-fragment run-on under a 140-character work-pane line).

(3) The graduation notice is delivered by self.notify (it always was -- library_screen.py's _apply_library_onboarding_evidence) and now fires only on STARTER -> GRADUATED. It used to greet a returning, already-populated profile on its first source read, which is what the seeded profile showed. The decision it drives moved into its own _apply_graduation_notice, called before the presentation legs that paint the carrier.

Deviation worth naming: the in-canvas #library-lifecycle-status line was KEPT. It is now raised only by the same STARTER -> GRADUATED transition, so the observed noise is gone; removing the carrier outright would delete _acknowledge_library_destination_change across two controllers and six test modules for no further user-visible gain. If the reviewer wants a toast and nothing else, that is a clean follow-up.

Nine existing tests that pinned the old rule were extended, not deleted (a starter=True seed on _new_library_onboarding_app, and a real STARTER -> GRADUATED transition in test_library_entry_compose_once.py). Files: tldw_chatbook/Widgets/Library/library_notes_canvas.py, library_note_work_pane.py, library_notes_add_from_files_canvas.py, tldw_chatbook/UI/Screens/library_screen.py, Tests/UI/{test_library_crit8_polish_shell,test_library_shell,test_library_entry_compose_once}.py, Docs/User_Guide/library.md, Docs/User_Guide/library/notes.md.
<!-- SECTION:NOTES:END -->
