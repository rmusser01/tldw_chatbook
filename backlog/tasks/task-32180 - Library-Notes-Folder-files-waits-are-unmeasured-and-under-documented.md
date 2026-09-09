---
id: TASK-32180
title: 'Library Notes: Folder files waits are unmeasured and under-documented'
status: Done
assignee: []
created_date: '2026-09-09 09:18'
updated_date: '2026-09-09 17:47'
labels:
  - library
  - notes
  - critique-notes-2026-09
  - rider
  - file-notes
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rider from the Notes critique fix wave (plan
Docs/superpowers/plans/2026-09-09-library-notes-critique-wave.md); raised in
the task review of task-32055/task-32121. Three loose ends in Folder files'
folder-change waits: the slow-wait "still working" busy row has no pinned
geometry test below 120 columns; the pre-link **Use \<folder\>** button only
ever reads the legacy `notes.sync_directory` config key, not any modern
equivalent; and the invariant that every re-entrant root change abandons the
previous one synchronously (so the previous scan's lock is always released
before the new one starts) lives only in the call graph between
`_abandon_root_change_task` and its callers, with no assertion or comment
recording it for the next reader.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The slow-wait busy row's layout is pinned in a test at 60 columns
- [x] #2 The pre-link **Use \<folder\>** button also honours the modern
  config key, or the guide explicitly names the legacy
  `notes.sync_directory` key it reads
- [x] #3 `_abandon_root_change_task` carries an assertion or comment
  recording the synchronous-abandon invariant
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Measure the slow-wait busy row at 60 columns (done: 'Choose another' runs 11 cells off-pane -- the four optional folder-row buttons never got the row's `min-width: 0` rule, so Textual's 16-cell Button default applies to all of them).
2. Failing geometry test at 60x24: every visible control in the busy row is on-pane and exactly one reads 'Cancel'.
3. Fix at the CSS seam the row already has; drop the controls a running folder change makes dead (Details shows the wait line, Change... is disabled while Choose another does the same job).
4. `Use <folder>` reads the modern `[file_notes] root` first and falls back to the legacy `[notes] sync_directory`; guide names both.
5. Comment on `_abandon_root_change_task` recording the re-entrant-abandon invariant and its honest limit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Three loose ends from the wave-1 wait work, one of which was a real off-pane defect.

AC#1 -- the busy row at 60 columns. Measured before writing anything: row width 60, and `Choose another` sat at x=55 with width 16, i.e. 11 cells past the right edge. Two causes, both fixed at the seam that already existed:
- Textual's Button default is `min-width: 16`, and the folder row's `width: auto; min-width: 0` rule only ever listed `#file-notes-root-details` and `#file-notes-choose-root`. The four optional controls (structural Cancel, Keep waiting, Choose another, Use folder) were added to that same rule -- "Cancel" no longer reserves 16 cells for six characters.
- While a change runs the row now belongs to its own three controls: `details.display` and `choose.display` are False in the wait branch. Neither was doing anything -- Details opens a dialog rendering `_root_status_detail`, which during a wait IS the line the user is already reading, and Change... is disabled for the whole transition while Choose another does its job. Deleting two dead controls, not shrinking live ones.
Result at 60 columns: `Changing folder... · <n> entries so far  Cancel  Keep waiting  Choose another` = 58 of 60 cells, status middle-elided, exactly one Cancel.

AC#2 -- `_configured_sync_folder` now reads `[file_notes] root` (the key this mode writes on every successful change) before falling back to the legacy `[notes] sync_directory`; both still go through `validate_existing_absolute_directory`. The guide names both keys explicitly, so the AC is met on either branch.

AC#3 -- comment, not an assertion, and it says why: the invariant is that every early end of a folder change funnels through `_abandon_root_change_task` (three call sites) and nothing else releases the previous scan's `_service_lock`; but it is NOT synchronous. Textual's exclusive-worker cancel delivers CancelledError on a later loop turn, so a re-entrant change can enter `_scan_for_root` while the old scan still holds the lock. The bounded poll-until-this-attempt's-flag acquire covers that overlap. An assertion claiming the previous attempt had finished would fire in normal use, and the docstring says so.

Tests: `test_the_slow_wait_row_keeps_every_control_on_pane_at_60_columns` (visible labels, every region inside the row, `_busy_row_cancel_labels == ['Cancel']`) and `test_use_folder_offers_the_modern_file_notes_root` (modern wins, legacy still falls back). Both RED first -- the first failed with `['Details', 'Change...', 'Cancel', ...]`.

Live: caps 03 / 04 (60x24 busy row at t+4s and t+12s, driven by starting the change at 235 and resizing down), 05 (timeout copy back with Details/Change...), 06 (`Use file_notes` from the legacy key, on-pane at 60 columns).

Deferred, deliberately: the `Use <folder>` button itself is still `width: auto`, so a very long folder name can overflow a narrow row. Out of this AC's scope (the busy row) and no evidence it bites.

Files: tldw_chatbook/Widgets/Library/library_file_notes_workspace.py, Tests/UI/test_library_notes_riders_r_file_notes.py, Docs/User_Guide/library/file-notes.md.

REVIEW ROUND 1. The 60-column fix was only half done: the wait branch kept `status.set_class(self._root is None, "-empty-root")`, so a folder change started from the UNLINKED empty state gave the busy line task-2850's `width: auto` hug -- it took 39 of 60 cells and `_fit_root_status` never elided (it measures `content_region.width`, which auto-width makes the full text). Keep waiting ended at 65 and Choose another at 83, both off-pane. That is exactly the defect AC#1 names, in the state task-32173 had just made reachable for the first time; my geometry test only covered the linked start. Fixed by `status.set_class(False, "-empty-root")` in the wait branch -- the same exclusion the `_root is None` branch already applies to its own reason line -- and the test is now parametrised over both starting states. RED for `[unlinked]` first: `Left contains 2 more items, first extra item: ('Keep waiting', Region(x=49, y=6, width=16, height=1))`.

Guide correction in the same round: the layout tour claimed the pre-link rail has a collapse grip. It does not -- `_sync_body_panes` gates `library_grip` on a linked root, deliberately, because an ungated grip would paint into the otherwise-empty compact body and break task-32173 AC#2. The sentence now says the grip arrives with the folder.

DEFERRED, no action taken: `Use <folder>` can still offer a folder that has just refused to link. `_configured_sync_folder` validates that the path is an existing absolute directory, not that this workspace can actually acquire it -- so after a failed link the button re-offers the same folder. Harmless (pressing it simply reports the same reason again) and out of this task's ACs; worth a rider if it is ever seen live.
<!-- SECTION:NOTES:END -->
