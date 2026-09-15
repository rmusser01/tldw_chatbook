---
id: TASK-32295
title: >-
  Two notes_create deep-link tests in test_library_shell.py never reach their
  deep link: both die waiting for the flat-list id #library-notes-row-0 while
  the canvas renders the folder tree
status: Done
assignee: []
created_date: '2026-09-10 20:03'
updated_date: '2026-09-11 10:45'
labels:
  - library
  - notes
  - tests
  - tech-debt
  - critique-notes-2026-09
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two Library shell tests are red on dev and have been for at least as long as
task-32245's fix wave: they die in the shared `_open_note_editor` helper before
reaching the deep link they exist to cover, so the notes_create deep-link
behaviour they name is currently unverified on every run.

Confirmed pre-existing, not a regression: task-32245 ran the same selection on
its branch AND on a detached `origin/dev` worktree (f6903baceb) and got an
IDENTICAL FAILED name set both times -- 7 passed / 2 failed on each.

Node ids:
- Tests/UI/test_library_shell.py::test_library_shell_notes_create_deeplink_reentry_resets_stale_editor_state
- Tests/UI/test_library_shell.py::test_library_shell_note_flush_on_notes_create_deeplink_saves_before_switching

Assertion (identical for both, raised from `_wait_for_selector`,
Tests/UI/test_library_shell.py:3812):

  AssertionError: #library-notes-row-0 never mounted within 30.0s (1146 polls).
  Visible text: ... Library notes . Library database . Ready . Next: Create a
  note or add from files. Notes (2) ... Select a note to edit it here. ...
  New folder > Unfiled   Q3 retro . 9w   Reading list . 9w

The shell is healthy in that dump -- both seeded notes are present and the
canvas has settled. `#library-notes-row-0` is a FLAT-list row id, and the
canvas is rendering the folder tree, which indexes its note rows across its
folder rows; the same flat-vs-tree id trap is already recorded in
`backlog/docs/lessons-testing-evidence.md` ("A widget id that becomes
conditional must be reconciled across all of Tests/", PR-wave 2026-09-09),
which reconciled other sites but not these two.

Repro: `pytest Tests/UI/test_library_shell.py -q -p no:randomly -k "nav_context
or deeplink or notes_create or entry_focus"` -> 7 passed, 2 failed on dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both node ids pass on dev without weakening what they assert -- each still drives its notes_create deep link and checks the state it names
- [x] #2 The note the helper opens is selected through an id or marker that is correct in BOTH the flat list and the folder tree, so the pair cannot silently stop running again when the presentation changes
- [x] #3 Any other Tests/ site waiting on #library-notes-row-0 while the tree can be active is reconciled in the same pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Run both node ids on the branch and on a detached `origin/dev` worktree.
2. If they are already green, prove WHY and check AC#3's sweep; do not
   re-repair what is repaired.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Already repaired on dev; verified rather than re-fixed.

Both node ids pass on `origin/dev` ff2dc03145 in a detached worktree (2 passed,
8.9 s) and on this branch (2 passed, 7.0 s). The repair landed with task-32175:
`_open_note_editor` waits on `.library-notes-row` (the class BOTH presentations
carry) and then selects by identity through `_wait_for_note_row(note_id)`, which
matches a row's `note_id` attribute rather than a position or a
`#library-notes-row-N` id -- which is exactly what AC#2 asks for, in both the
flat list and the folder tree.

AC#3 sweep: the only remaining `#library-notes-row-0` wait under `Tests/` is
`Tests/UI/test_library_notes_wave_list.py::test_flat_rows_render_the_same_single_
line_age`, which builds a `_CanvasApp` with NO tree projection on purpose -- the
flat fallback is the thing under test there, so the flat id is correct and the
tree cannot be active. (`Tests/UI/test_library_canvas_scoped_sync.py:153` is a
comment recording the same trap.) Nothing to reconcile.

No code change. Files: none.
<!-- SECTION:NOTES:END -->
