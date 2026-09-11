---
id: TASK-32102
title: >-
  Library structural waits: follow-ups from the task-32055 review (timer
  retention, File Notes Cancel timing, narrow back toggle, cancelled-import
  refresh)
status: Done
assignee:
  - '@claude'
created_date: '2026-09-08 22:42'
updated_date: '2026-09-10 17:06'
labels:
  - library
  - file-notes
  - skills
  - cleanup
  - critique-8
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the task-32055 reviews (PR #2524): the patience `set_timer` in `_begin_library_structural_wait` is neither retained nor stopped; the File Notes Cancel button is revealed at t=0 while the status line mentions Cancel only after 3 s; `#file-notes-back` (narrow-view toggle) does not cancel a running folder change (the 30 s deadline is the only backstop); a cancelled skill import leaves `refresh_sources=False` although its copy says to check the skills list; `except asyncio.CancelledError: if not wait.cancelled: raise` tests the wait flag rather than `change.cancelled()`; the wait branch of `_update_root_surface` skips the -warning/-offline resets; the cancelled copy is written to a canvas being torn down on the exit-triggered path. Rider from the critique-8 fix wave reviews (plan Docs/superpowers/plans/2026-09-08-library-crit8-wave.md; wave PRs #2519 #2523 #2524 #2525 #2528 #2531).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each of the seven items is fixed or closed with a recorded reason (2 of 7 fixed here; the other 5 are implemented on task-32180 / PR #2557 and land with it -- tick on that merge)
- [x] #2 The File Notes Cancel affordance and its copy appear at the same moment (lands with #2557: implemented on task-32180 / branch fix/library-notes-r-file-notes at 203f4dc633, tick on that merge)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Scope note: PR #2557 (task-32180, Notes critique wave 2) is rewriting the File
Notes slow-folder-change row, so the four items that live inside
`library_file_notes_workspace.py` are handed to it (Cancel reveal timing,
`#file-notes-back` cancelling a running change, the `change.cancelled()`
condition, the `_update_root_surface` wait-branch resets, and the cancelled
copy written to a torn-down canvas). This task fixes the two that live
elsewhere:
1. Retain the patience `set_timer` per owner in `library_screen.py` and stop
   it when the wait ends or is replaced, so a settled wait cannot repaint a
   surface it no longer owns.
2. A cancelled skill import must set `refresh_sources=True` in
   `_cancelled_outcome` -- its own copy tells the user to check the skills
   list, which the receipt never refreshed.
TDD, one new test file (`Tests/UI/test_library_crit8_wait_riders.py`) so the
concurrent PR's own tests in `test_library_crit8_waits.py` do not conflict.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Split by file owner. Five of the seven items live inside
`Widgets/Library/library_file_notes_workspace.py`, which PR #2557
(task-32180, branch fix/library-notes-r-file-notes, Notes critique wave 2)
owns. They are implemented there, not here, at commit 203f4dc633 with
RED/GREEN tests in `Tests/UI/test_library_notes_riders_r_file_notes.py`:

- **Cancel timing** (`test_cancel_appears_only_once_the_wait_admits_it_is_slow`):
  the File Notes wait line never carries "· still working · Cancel" --
  task-32121 deliberately dropped that suffix on this surface. The Cancel
  BUTTON is what carries the word, and its reveal is now gated on the same
  3 s slow flag that Keep waiting / Choose another already use, so no new
  clock was introduced.
- **`_update_root_surface` wait branch**
  (`test_the_wait_line_carries_no_tint_from_the_state_it_replaced`): the
  wait branch now clears `-empty-root` as well as `-warning`/`-offline` --
  none of the three classes survive the wait.
- **Cancelled copy on a torn-down canvas**
  (`test_leaving_during_a_folder_change_announces_the_cancellation`): the
  exit-abandon path notifies instead of writing to the canvas, covering
  Escape / back / rail switch through `_flush_active_file_notes` →
  `cancel_structural_wait(leaving=True)`. In-place exits (the Cancel button,
  Choose another) keep the row copy, because that surface is still on
  screen.
- `#file-notes-back` cancelling a running folder change and the
  `if not wait.cancelled` → `change.cancelled()` condition live in the same
  file and are that PR's to land.

AC#2 stays unticked here: it lands with #2557. **Fix round 1 (PR #2569 review, finding 7): AC#1 is now unticked and the task back In Progress for the same reason** -- it covers all seven items, four of which exist only on that peer branch, so ticking it here would have left them untracked if #2557 slipped. Both ACs tick, and this task goes Done, on that merge; the two items owned here are complete and shipped in PR #2569.

Fixed here, both outside that file:
- **Patience timer retained and stopped** (`library_screen.py`). The one-shot
  `set_timer` was fire-and-forget, so a wait that settled inside the three
  second window still repainted a surface that had moved on. Timers are now
  kept per owner in `_library_structural_wait_timers` and stopped both by
  `_end_library_structural_wait` and by a second `_begin_...` for the same
  owner (a restarted folder change must not inherit the old clock).
- **A cancelled skill import refreshes the list it names**
  (`library_skill_import_controller.py`). `_cancelled_outcome` shipped
  `refresh_sources=False` while its own copy says 'check the skills list
  before retrying' -- and the reason that copy exists is that the worker
  thread may have landed the import anyway, so the list behind the receipt
  was the pre-import one.

Tests live in a NEW file, `Tests/UI/test_library_crit8_wait_riders.py`,
rather than `test_library_crit8_waits.py`, because the latter is PR #2557's
surface. The wait-registry tests bind the two unbound screen methods onto a
three-attribute host, so they need no app or event loop.

No live verification: the timer fix is invisible by construction, and
driving a >3s import to a Cancel needs an artificially blocked scan, which
is what `test_library_crit8_waits.py`'s own harness already does (20 passed
with the timer change).

Files: `tldw_chatbook/UI/Screens/library_screen.py`,
`tldw_chatbook/UI/Library_Modules/library_skill_import_controller.py`,
`Tests/UI/test_library_crit8_wait_riders.py`,
`Docs/User_Guide/library/skills.md`.

**Closed 2026-09-11 on evidence:** the five File Notes items (Cancel affordance timing, `-empty-root` cleared alongside `-warning`/`-offline`, exit-abandon notify via `cancel_structural_wait(leaving=True)`, and the two others) shipped with the peer's PR #2557 (task-32180, commit 203f4dc633, merged 2026-09-11 00:36Z); the two items owned here shipped with PR #2569 (task review + fix rounds). All seven are now on dev.

<!-- SECTION:NOTES:END -->
