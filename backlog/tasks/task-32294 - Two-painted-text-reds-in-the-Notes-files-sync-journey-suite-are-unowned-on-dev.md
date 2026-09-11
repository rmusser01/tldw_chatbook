---
id: TASK-32294
title: >-
  Two painted-text reds in the Notes files-sync journey suite are unowned on
  dev
status: Done
assignee: []
created_date: '2026-09-10 12:55'
updated_date: '2026-09-11 10:45'
labels:
  - library
  - notes
  - critique-notes-2026-09
  - rider
  - file-notes
  - tests
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/UI/test_library_notes_files_sync_journey.py` has two tests that are red
on clean `dev` and belong to no open task, so nobody is watching them: a
reader who runs that file while working on Library ▸ Notes cannot tell a
regression they just caused from the reds that were already there. Both
assert on painted text or on what the compositor actually shows, so both are
either a real paint regression or a stale expectation — which of the two is
the work.

Found while landing PR #2557 (wave-2 Notes fix wave), where they had to be
confirmed as pre-existing before the PR's own results could be read. They are
not caused by that PR: the same two names, with byte-identical assertion
output, fail on the branch and on the `dev` tip it was merged from.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 `test_lasting_setup_keeps_server_unavailable_copy_painted[size0]`
  passes, or the assertion is corrected to what the screen is supposed to
  paint with the reason recorded
- [x] #2 `test_folder_files_and_session_git_use_supported_40x20_navigator`
  passes, or the assertion is corrected to what a 40x20 terminal is supposed
  to show with the reason recorded
- [x] #3 Whichever of the two turns out to be a product defect rather than a
  stale expectation is fixed at its source, not by relaxing the assertion
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce both, and for each decide paint regression vs stale expectation
   from the measured screen, not from the assertion text.
2. Repair to the truth at equal or better strength; file the product half.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Both turned out to be stale expectations; one of them sits on top of a real
product gap, now filed.

AC#1 `test_lasting_setup_keeps_server_unavailable_copy_painted[size0]` -- NOT a
paint regression. Measured at 120x36: the reason renderable is fully on screen,
WRAPPED across two rows inside its own 45-cell box ("... server sync-folder
capability" / "not installed"), so `"capability not" in painted` failed on the
line break while `"server sync-folder"` passed. At 60x20 the wrap falls
elsewhere and all three substrings hit -- which is why only `[size0]` was red.
Replaced by a strictly stronger check: `_painted_widget_text` crops the
compositor strips to the widget's own region and collapses the wrap, and the
test asserts the WHOLE reason sentence appears there. Both sizes green.

AC#2 `test_folder_files_and_session_git_use_supported_40x20_navigator` -- the
40x20 expectation was wrong, and the reason is a product gap. Measured:
`#file-notes-work` resolves to `Region(x=40, y=7, width=1, height=12)` -- the
work pane is off the right edge of a 40-column screen -- so Session Git is
mounted and never composited. Pressing it opens the route (its Back control
reports `display` and takes focus) while the painted screen does not change at
all. Reopening the pane through `#library-file-notes-items-grip` (which IS
composited) gives it `Region(x=10, y=7, width=30, height=12)`, but the button
then lands on row 19, below the pane's 12 rows. The test now asserts what 40x20
really shows (navigator painted, authority composited, Session Git mounted but
not composited, the grip on screen) and that the ROUTE still opens and takes
focus.

AC#3 -- the product half is filed as task-32452 rather than fixed here: making
the Folder files work pane claim the screen at the narrow floor is a stage/
priority design change for that surface, not a test-health repair. The
assertion was corrected WITH the measurement and the pointer to that task, not
relaxed.

Suite check: `Tests/UI/test_library_notes_files_sync_journey.py` whole file =
1 failed / 29 passed; the one failure
(`test_database_notes_import_once_journey_is_painted_focused_and_retained[size1]`)
fails identically on dev.

Files: `Tests/UI/test_library_notes_files_sync_journey.py`.
<!-- SECTION:NOTES:END -->
