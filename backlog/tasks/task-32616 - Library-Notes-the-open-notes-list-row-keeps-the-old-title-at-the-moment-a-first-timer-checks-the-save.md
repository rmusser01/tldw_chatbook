---
id: TASK-32616
title: >-
  Library Notes: the open note's list row keeps the old title at the moment a
  first-timer checks the save
status: Done
assignee: []
created_date: '2026-09-15 06:41'
labels:
  - library
  - notes
  - critique-4
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 (dev 77eb2601a6), assessor A P2 and assessor B D20 (PARTIAL), persona Jordan, create workflow. Both assessors hit it independently; A names it the emotional valley of the whole first journey.

What happened. Jordan types a title, the editor heading reads 'My first note' and the status reads 'Saved 22:50' -- and the only durable artefact on screen, the row in the list beside them, still reads 'Untitled · now' (A caps 06, 07). It corrects the moment Escape leaves the editor (A cap 08). B saw the same shape later in the journey: after changing a title to a new value the list row still showed the old one while Info was open (B cap 58). For a first-timer the question is never 'did the widget update', it is 'is my writing safe', and the screen answers it two ways at once.

Cause PROVEN and pinned: notes.md documents the refresh as deliberately skipped while the title field holds focus. The reasoning is sound for a rename; applying it to a note whose title has never been anything, and to a row that is the user's only save receipt, is the part both assessors dispute. This is a design decision to revisit, not a bug to fix blind.

Compounding, same pane, same moment: two 'Next:' instructions disagree with each other -- the list's 'Library notes · Ready · Next: Create a note or add from files.' beside the editor's 'Empty note — type to keep it · Next: Start typing.' (A cap 04) -- and after typing the editor's still reads 'Next: Start typing.' (A cap 05). Open riders 32513/32514 hold the twice-painted save state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The currently open note's own row reflects its committed title, or is marked as the open note so the stale label reads as a pointer rather than a contradiction
- [x] #2 The focus guard that skips the refresh still holds for every other row
- [x] #3 One pane never shows two Next instructions that disagree
- [x] #4 The decision is recorded either way, with the reasoning, so critique 5 does not re-litigate it
<!-- AC:END -->

## Implementation Plan

1. Trace the producer: why the row stays stale after a rename save.
2. Repaint the list on a genuine rename only, without weakening the task-32062
   focus guard.
3. Re-derive the "two Next instructions" claim live before fixing it.
4. Record the decision either way.

## Implementation Notes

**AC#1/AC#2 -- the producer, not the renderer.** A save already patched the
list CACHES (`_patch_library_note_list_from_session`: the flat records, the
tree branch slices, the filter window) and then painted nothing, so the row
beside the editor kept the pre-rename label until the next sync -- in
practice, until Escape. The patch now RETURNS whether the label it was
rendering actually changed, and `_apply_library_note_saved_presentation` runs
one `_sync_library_canvas(self, "notes")` when it did.

The task-32062 focus guard is untouched (AC#2). It lives in
`LibraryNotesCanvas.sync_state` and is measured per canvas: the WORK pane
still skips its rebuild while the title or body has focus, and the Items pane
is a sibling canvas that recomposes as it already does for every other sync,
with `canvas_sync`'s `notes_editor_owned` branch restoring its scroll offset
and never touching focus. The pin asserts both halves at once -- the row
carries the new title AND the title `Input` is the same object, still focused,
still holding its text.

Gated on a genuine rename rather than on every save, so a body-only autosave
(one per debounce tick) costs the tree nothing.

**AC#3 -- one half was not a defect.** Re-derived live at dev 3b26c66ce0:

- *Two disagreeing Next instructions: REPRODUCED.* Verbatim, both painted at
  once: "Library notes · Ready · Next: Create a note or add from files."
  beside "Saved · Next: Start typing." The list's instruction is advice for a
  reader with nothing open; with a note open beside it, it is advice against
  what they are doing. Fixed by standing the list's clause down while the work
  pane has one (`LibraryNotesListState.note_open`).
- *"After typing, the editor still reads Next: Start typing": NOT
  REPRODUCED.* Measured through the real screen: after a TITLE-only edit the
  body really is empty, so "Start typing." is the true next step, and the line
  becomes "Keep editing; changes save automatically." the moment the body has
  words. A first implementation "fixed" this by updating the authority Static
  inside the task-32062 skip; that change PASSED with the fix reverted -- the
  proof it was fixing nothing -- and was removed rather than shipped
  unpinned, since it also partly undoes a deliberately documented ceiling
  (the skip's own "Known ceiling" comment).

**The repaint cost, COUNTED (review round 1).** The earlier note said only
that a body-only autosave changes no row label, which is true but left the
title path unstated -- and `handle_library_note_title_changed` arms the same
`_schedule_library_note_autosave` debounce the body handler does, so a title
edit also saves once per tick. The question is not which path saves but which
SAVE repaints. Counted at `_sync_library_canvas(self, "notes")` in
`library_notes_controller`, driving the real screen:

* three body-only saves -> **0** Items-pane syncs;
* one save that changes the title -> **exactly 1**.

Reverting the `if renamed` gate turns the first number into 3, which is what
wires the count to the fix. Replaces the `inspect.getsource` check for
`"return title_changed"`, which could not show the asymmetry that IS the cost
argument.

**AC#4 -- the decision, recorded.** The focus guard stays, narrowed to what it
was actually for. The reasoning: the guard exists so a refresh never rebuilds
the field under the reader's hands. It was never about the ROW, and applying
it to the row made the only durable artefact on screen contradict the heading
and the save status next to it at the exact moment a first-timer checks
whether their writing is safe. Repainting a sibling canvas rebuilds no field,
moves no focus, and costs one recompose per rename. Recorded in
`Docs/User_Guide/library/notes.md`, which previously documented the old
behaviour as intentional.

**Files.** `UI/Screens/library_screen.py`,
`UI/Library_Modules/library_notes_controller.py`,
`Widgets/Library/library_notes_canvas.py`, `Library/library_notes_state.py`,
`Tests/UI/test_library_notes_w5_import_preview.py`,
`Docs/User_Guide/library/notes.md`.
