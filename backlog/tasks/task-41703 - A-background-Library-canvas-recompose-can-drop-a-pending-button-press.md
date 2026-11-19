---
id: TASK-41703
title: "A background Library canvas recompose can drop a pending button press"
status: Done
assignee: ['@claude']
created_date: 2026-11-18 04:24
updated_date: 2026-11-19 16:39
labels:
- follow-up
- ui
- library
priority: medium
references:
- tldw_chatbook/UI/Library_Modules/canvas_sync.py
- tldw_chatbook/UI/Screens/library_screen.py
- Tests/UI/test_library_shell.py
dependencies:
- TASK-41532
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Every Library canvas `sync_state` recomposes all of the canvas's children. Most presses caught by such a recompose are still delivered, just late. A press **can be** lost when it is still below the canvas child the recompose removes at the moment that child stops taking messages; the user's click then does nothing. TASK-41532's control lost 1 press in 48 trials with an unawaited sync, and 5 in 48 with an awaited recompose.

Found by TASK-41532 on the Media type-strip opener in `test_library_media_complete_facets_commit_during_initial_recovery[loading-All]`. The opener was pruned before its press reached the handler, and focus ended on the replacement type filter.

Lead, unverified: Media's empty-list entry fallback (`_library_media_empty_list_fallback_target`) runs while the 2 s entry window is armed.

Until this is fixed, the test works around it by re-pressing the opener (`Tests/UI/test_library_shell.py`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A pilot test holds a press in a widget below the canvas child (for example the button), releases it once that child has stopped taking messages, and shows the press handled; red N of N before the fix
- [x] #2 The notes name what syncs the Media canvas while page 1 is held back in `[loading-All]`
- [x] #3 The re-press workaround in `test_library_media_complete_facets_commit_during_initial_recovery` is removed once the press can no longer be lost
<!-- AC:END -->

## Implementation Plan

1. Instrument the `[loading-All]` test (temporary) to record every Media canvas sync and recompose after the opener press, with its stack, and name the trigger (AC#2).
2. Find the seam every Library canvas recompose routes through (`PostRecomposeCallback.recompose`) and fix the drop there: Textual removes the old children top down, so a canvas child stops taking messages first and a `Button.Pressed` still queued below it is dropped where it would bubble into the closing parent.
3. Add a pilot test that gates a press in the button, recomposes the canvas, releases the press once the canvas child is closing, and asserts the handler ran; red N of N before the fix.
4. Remove the re-press workaround from `test_library_media_complete_facets_commit_during_initial_recovery`.

## Implementation Notes

Two causes, one in the product and one in the test.

**The product defect (AC#1).** Textual removes a recompose's old children top down. A canvas child (the Media toolbar row) processes its `Prune` and stops taking messages first; only then are the nodes below it pruned. A button below it that then handles a queued `Button.Pressed` finds `is_parent_active` False, and `MessagePump._on_message` does not bubble the press, so the click does nothing. Every Library canvas recompose goes through `PostRecomposeCallback.recompose` (`Widgets/Library/library_canvas_sync.py`), so the fix is there: before the removal, each node from a button up to (not including) the canvas child gets a per-instance `_on_message` wrapper. After its own dispatch, if the message is an unstopped `Button.Pressed` and its parent is closing, it bubbles the press to the canvas (`Message._bubble_to`), which outlives its own recompose and passes it on to the screen. The forwarded press reaches the canvas and above only: handlers on the removed nodes it skips do not run, a canvas child's own `@on(Button.Pressed, ...)` included (when `LibraryNotesCanvas` recomposes, that is `LibraryNoteImportCanvas`'s `#note-import-check` handler and `LibraryNotesAddFromFilesCanvas._button_pressed`). Before, the press was lost there as well; the docstring and the lesson say so (review r17ls-rev1). Nothing is forwarded past the exit fence or when the canvas itself is being removed. The wrapper is installed on the old children only, so the new children carry no overhead.

- **Test.** `Tests/Library/test_library_notes_canvas_pending_lookup.py::test_a_press_queued_below_a_removed_canvas_child_still_reaches_the_screen`, a bare Textual app with a `PostRecomposeCallback` canvas in the Media toolbar's shape (canvas, row, button). The button's pump is held with `call_later` on an awaited event, the press is queued behind it, the canvas recomposes, and the press is released once the row is `_closing`. Before the fix: 5 of 5 failed, "the press below the removed toolbar was dropped". After: 5 of 5 passed (serially, load 30-44).
- The file is a whole-file target in the derived-artifacts fast lane, so no census entry is needed (about 2 s).

**What synced the Media canvas while page 1 was held back (AC#2).** Three syncs, all before the press: the page-1 request's (`LibraryMediaBrowseController.request`), the facet request's (`request_facets`), and the facet completion's (`_load_facets`). Each goes `_sync` -> `_sync_library_media_browse_state` -> `_sync_library_canvas(..., "media")` -> `LibraryMediaCanvas.sync_state` -> `refresh(recompose=True)`. The first two usually share one recompose and the facet completion takes a second. A probe logged every canvas sync, `refresh(recompose=True)`, recompose start and end, and every `App._prune` that took the opener, from 1.5 s before the press to its handler: 84 runs with today's wait and 48 with the CI-time one showed nothing between the press and the handler.

The CI failure (run 43917600414, at 5e3c8b2bd4) came from the test's wait, not from a background sync. That wait read only `not canvas._recompose_required`, and Textual's `_check_recompose` clears that flag when a recompose starts. A probe that checked the CI-time wait every loop turn found it true while the facet completion's recompose was in flight in 24 of 24 trials. In that window the old opener is still queryable but `_pruning`, so `display` reads False and `Button.press()` returns without posting anything (24 of 24 never handled). That matches TASK-41532's round-1 diagnostic: `display=False, pruning=True, parent=None`, with focus on the replacement opener (the empty-list entry fallback had focused the opener, and the recompose kept focus on its same-id replacement). cc58dd5004 (#3533) has since made the wait also require `not recompose_in_flight`, no pending post-recompose callback, and a disarmed entry window, which closes that window. With the CI-time wait, 15 of 48 presses still landed with the 2 s entry window armed (the lead in the description), and none of them was lost.

**Workaround removed (AC#3).** The re-press loop and its pump drain are gone from `test_library_media_complete_facets_commit_during_initial_recovery`; the test presses once and waits for the chooser. The three cases, 8 repeats each at `-n 4`: 24 of 24 passed (load 28-30). A control that forced `await canvas.recompose()` right after the press (temporary, reverted) passed 48 of 48 with the fix and 48 of 48 without it at load 30-36, so the press nearly always reaches its handler first in the app; the pilot test above is the red/green evidence for the drop.

**Covering runs.** `Tests/Widgets/Library`, the pending-lookup file, `test_library_canvas_scoped_sync.py`, `test_library_canvas_sync_gaps_t21242.py`, `test_library_canvas_sync_defects.py`, `test_library_focus_never_lands_on_a_removed_widget.py`, `test_library_notes_folder_navigator.py`, `test_library_notes_w4_editor.py` and the TASK-41715 files, at `-n 4`: head and base fail the same nodes (89, the local `RecoveryRequired` trap and its fallout) except three. Two were `test_library_notes_w4_editor.py` nodes whose `_QueueHost` fake had no `walk_children`; it now returns `[]`, and both pass. The third, `test_repair_continuation_reaches_every_candidate_and_restarts_after_mutation[size0]`, passed 2 of 2 serially (it failed at base in an earlier run).

**Files.** `tldw_chatbook/Widgets/Library/library_canvas_sync.py`; `Tests/Library/test_library_notes_canvas_pending_lookup.py`; `Tests/UI/test_library_shell.py`; `Tests/UI/test_library_notes_w4_editor.py`; `backlog/docs/lessons-textual.md` (new entry).
