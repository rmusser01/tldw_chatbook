---
id: TASK-32458
title: >-
  Library Notes: a late backlinks result kills the whole app when the work pane
  recomposed underneath it
status: Done
assignee:
  - '@claude'
created_date: '2026-09-11 09:55'
labels:
  - library
  - notes
  - bug
  - crash
  - critique-notes-2026-09
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Opening a note and then clicking inside it before the "Linked from" lookup
answers terminates the whole application. Hit twice while verifying the wave-3
editor-key group on dev 4a14b3f36f, both times fatally: the first occurrence
killed the tmux session outright before I was capturing, the second is on
record with its full traceback.

    NoMatches: No nodes match '#library-note-work-authority' on
    LibraryNoteWorkPane(id='library-note-work-pane',
    classes='library-notes-mode-editor library-adaptive-reader-work')

    Worker ERROR name='_load_library_note_backlinks'
      group='library_note_backlinks'
    library_notes_controller.py:3416  self._apply_library_note_presentation_state()
    library_notes_controller.py:1635  canvas.apply_session_state(
                                          self._library_note_presentation_state())
    library_notes_canvas.py:2440      authority = self.query_one(
                                          f"#{self.authority_id}", Static)

Cause PROVEN, and narrower than "a stale worker": the tail of
`_load_library_note_backlinks` already guards the note-identity half --

    if note_id != self._selected_note_id or self._library_notes_view != "editor":
        return

-- and there is no guard at all for the pane-identity half. The note is still
the same note and the view is still `"editor"`, so the guard passes; what
changed underneath is the mounted subtree. `apply_session_state` reaches for
`#library-note-work-authority` on a work pane that is mid-recompose and has not
re-mounted it yet, which is the very race `_apply_post_compose_state` is
commented as handling for the sibling path ("``sync_state`` mutates the fields
and only SCHEDULES the rebuild"). `apply_session_state` has no equivalent
check.

The consequence is the severe part. This runs under `run_worker`, whose default
`exit_on_error=True` means an uncaught exception in the worker takes the app
down -- not a failed panel, not a toast, the whole process, with any unsaved
draft in any other tab going with it. The backlink lookup itself is already
defensive (its `await` is wrapped in `except Exception` and degrades to a
"couldn't check" status); it is only the paint that follows it that is not.

Introduced by **PR #2552** (task-32145, "Info lists the notes that link to the
open one"), which added the worker: `2eb7c55949` is the first commit containing
`_load_library_note_backlinks`, and PR #2552 is the merge that brought it to
dev. Everything below the worker in the traceback is older shared code that had
no asynchronous caller before this one.

Out of scope for the wave-3 editor-keys group (task-32246/32247/32252/32253/
32267/32268), which touched none of this path -- the crash reproduces on
unmodified dev code. **Assigned to the wave-3 backlinks-table group
(task-32186)**, which already owns this worker and its Info panel.

Evidence: captures from the wave-3 editor-keys live session on a seeded profile
at 235x52, `scratchpad/wave3-caps/editor-keys/18-crash.txt` (the error screen)
and `19-crash-work-authority.txt` (the full traceback, saved with
`capture-pane -S -400`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A backlinks result that arrives after the work pane recomposed, or after the open note changed, never raises and never paints onto the wrong note
- [x] #2 Covered by a test on the real worker hand-off, failing on the live behaviour before the fix
- [x] #3 Verified live by switching notes rapidly while backlinks are still loading, with the app still running afterwards
<!-- AC:END -->

## Implementation Plan

1. Reproduce the exact DOM shape (mounted work pane, children between removal
   and remount) through the real worker hand-off.
2. Guard `apply_session_state` itself rather than its callers.
3. Verify live by switching notes rapidly while the lookup is in flight.

## Implementation Notes

Fixed in `LibraryNoteWorkPane.apply_session_state`
(`tldw_chatbook/Widgets/Library/library_notes_canvas.py`), as the second half
of the early-out that method already had: when the pane is mounted but its
CHILDREN are between removal and remount, store the state and return instead
of `query_one`-ing a surface that is not there.

Not at the controller seam. An early return in
`_apply_library_note_presentation_state` also skips
`title_placeholder_only` and the stage-visibility and footer work that call
sits between, so the pane would come back from its recompose with stale
chrome; and several other callers reach `apply_session_state` directly and
would still raise. (A controller-level guard was tried on the test-health
branch, `dd1722ba01`, and reverted for exactly that.)

The worker's own note-identity guard was already correct and is untouched:
`note_id != self._selected_note_id or view != "editor"` returns before any
paint. What was missing is pane identity.

Dropping the paint is complete, not lossy: the state is stored before the
guard and `_apply_post_compose_state` re-applies it when the recompose mounts
the new children — pinned by the test, which recomposes afterwards and
asserts both backlink rows reach Info.

RED -> GREEN: `Tests/UI/test_library_notes_riders_backlinks.py::
test_a_backlink_result_landing_mid_recompose_does_not_kill_the_app` drives the
REAL worker (`_load_library_note_backlinks`) after emptying the work pane, so
the race is forced rather than waited for — deterministic where the live
symptom is roughly 1-in-3. RED 4 of 4 runs with the production exception
verbatim (`NoMatches: No nodes match '#library-note-work-authority' on
LibraryNoteWorkPane(...)`); GREEN 4 of 4, and 8/8 for the file.

Live (AC #3): seeded profile at 235x52, ~27 note rows clicked in rapid
succession with an Info toggle mid-run while lookups were in flight; app still
running, no traceback
(`scratchpad/wave3-caps/backlinks-table/live-rapid-switch.txt`).

Fixed on the wave-3 backlinks-table branch alongside task-32186, which owns
this worker's query.

Modified: `tldw_chatbook/Widgets/Library/library_notes_canvas.py`,
`Tests/UI/test_library_notes_riders_backlinks.py`.
