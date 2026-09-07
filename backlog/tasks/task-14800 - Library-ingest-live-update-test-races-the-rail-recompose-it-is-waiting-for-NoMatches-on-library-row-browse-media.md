---
id: TASK-14800
title: >-
  Library ingest live-update test races the rail recompose it is waiting for
  (NoMatches on #library-row-browse-media)
status: Done
assignee: []
created_date: '2026-08-09 17:30'
updated_date: '2026-08-11 03:01'
labels:
  - library
  - tests
  - flake
  - dev-baseline
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/UI/test_library_shell.py::test_library_shell_ingest_canvas_live_updates_without_manual_recompose`
is an intermittent failure on dev. Its third poll loop waits for the rail's
post-ingest `Media (N)` count to increment, but it reads the row with a bare
`screen.query_one("#library-row-browse-media", Button)` inside the loop body.
The count refresh it is waiting for is delivered BY a rail recompose, so the row
is transiently unmounted for one or more poll ticks — and `query_one` raises
`textual.css.query.NoMatches: No nodes match '#library-row-browse-media' on
LibraryScreen()` instead of retrying. The loop's own attempt budget
(`_INGEST_POLL_ATTEMPTS`, 500 x 20ms) is never reached; the test dies in ~4s.

Pre-existing and dev-side: the test was authored at `d54c7a252` (2026-07-09,
"feat(library): live ingest queue updates and post-ingest count refresh") and is
byte-identical on dev `f6911b37b`. Found while repairing the notes half of this
file after rebasing `feat/media-ingest-followups` onto dev `f6911b37b`
(task-3315 round 2), which touched neither this test nor any product code it
exercises.

Evidence:
- Failed once in a plain non-notes batch run (`-k "not note"`): 1 failed / 295
  passed.
- Passes 8/8 when run alone on an unloaded machine.
- Reproduced deliberately under 12-way CPU saturation: 1/6 then 1/1, both with
  the same `NoMatches` at the in-loop `query_one` (test file line 15709 at the
  time of writing).

Sibling observation, NOT reproduced and not necessarily the same defect:
`test_library_note_remount_restores_only_persistent_view_state[terminal_size0]`
was seen failing once in a full-file run during the same rebase triage, then
survived 26 subsequent runs (10 alone, 3 in its file-order neighbourhood, 6
under CPU saturation, 4 immediately after two deliberately-failing neighbours,
and 3 full notes-half runs). No failure text was captured. Recorded here only so
a future recurrence has a starting point; there is no evidence today that it
shares this cause. Note that `Tests/UI` already gets per-test `gc.collect()`
(`Tests/conftest.py` `_APP_MOUNTING_DIR_PARTS`), so `TLDW_TEST_GC_EVERY=1` is a
no-op here and the task-1468 app-cycle interference class is already excluded.

A minimal de-race (query tolerantly inside the loop, keep the budget as the only
failure signal) was applied on `feat/media-ingest-followups` so the file could be
proven green as a set; this task covers auditing the rest of the file for the
same in-loop-`query_one`-across-a-recompose shape and deciding whether the
pattern deserves a shared helper.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `test_library_shell_ingest_canvas_live_updates_without_manual_recompose` survives a loop under deliberate CPU contention without a `NoMatches` failure
- [x] #2 A failure of that test reports the real symptom (count never incremented, with the last observed label) rather than a `NoMatches` from a transient recompose window
- [x] #3 Every other poll loop in `Tests/UI/test_library_shell.py` that reads a widget which the awaited change recomposes is either tolerant of the gap or shown not to need it
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Enumerate every poll loop in Tests/UI/test_library_shell.py with an AST pass (a For/While whose body awaits pilot.pause/asyncio.sleep), not grep guesswork, so the coverage claim is checkable.
2. Flag the loops whose body performs a read that RAISES when the node is absent (query_one/get_widget_by_id/DOMQuery.first/subscript-of-a-call).
3. Classify each flagged loop by hand: does the change it awaits recompose the widget it reads?
4. For the genuine ones, route the poll through one shared tolerant helper (not N ad-hoc try/excepts) that keeps the attempt budget as the only failure signal and reports the last observed value.
5. Re-run the enumeration to prove only the classified-benign loops remain.
6. Mutation-check the helper: an unsatisfiable predicate must fail with the last observed value, never a NoMatches.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
AC#1/#2 shipped earlier on this branch (the third poll loop now queries tolerantly and reports the last observed label). This note covers AC#3, the audit of the rest of the file.

**Enumeration method (checkable, not asserted).** An AST pass over `Tests/UI/test_library_shell.py` (23k lines) collected every `For`/`While`/`AsyncFor` whose body awaits a pause (`pilot.pause`/`asyncio.sleep`) -- the operational definition of a poll loop -- and flagged those whose body performs a read that RAISES when the node is absent: `query_one`, `get_widget_by_id`, `DOMQuery.first/last/only_one`, and subscripting a call result (`list(query(...))[0]` raises IndexError in the same gap). **188 poll loops; 17 flagged.** Each flagged loop was then classified by hand against the real production seam.

**Classification.** Three were the genuine shape and are now tolerant:
- note autosave status (`#library-note-status`) -- bare `query_one` at the top of the wait for the 'Saved' flip;
- both 'Clear finished' arming polls (`#library-ingest-clear-finished`) -- the button is a child of `LibraryIngestQueuePanel`, and `_update_library_ingest_dynamic_regions` does `queue.refresh(recompose=True)` on every job tick, so it is transiently unmounted while jobs settle.

The other 14 are shown not to need it: 6 read no widget at all (`host.screen_stack[-1]`, the recording service's call list, the job registry's counts dict); 4 already query tolerantly under an `if matches:` guard (including the AC#1 fix and task-3315's); 2 are iterations rather than condition polls (the RAG result rows; the rail sections, whose toggle flips `body.display` IN PLACE with no recompose at all); 1 (the sync progress status) is proven un-raceable by the test's own next assertion, `status_widget_before is status_widget_mid_run`; and 1 (`test_library_media_entry_focus_survives_three_chained_recomposes`) drives its own recompose and then ASSERTS the row -- tolerance there would delete the regression it guards, so it is deliberately left alone.

**One shared helper, not N try/excepts.** `_wait_for_widget_state(screen, pilot, selector, predicate, *, what, attempts, interval)` sits beside the file's existing `_wait_for_selector`/`_wait_for_condition`: it queries tolerantly, keeps the attempt budget as the ONLY failure signal, and reports the last value it actually observed.

**Evidence.** Re-running the enumeration after the change leaves only the classified-benign loops (plus the helper's own tolerant loop). `Tests/UI/test_library_shell.py` in two disjoint halves: `-k note` **256 passed**, `-k 'not note'` **296 passed** (552 total). `test_library_ingest_canvas.py` **96 passed**; `test_library_ingest_structural.py` **22 passed**. AC#1 re-verified: the live-updates test looped 6x under deliberate 12-way CPU saturation, 6/6 green, no `NoMatches`. Mutation: pointing one converted call site at an unsatisfiable predicate fails with `AssertionError: first press never armed the button (last observed: 'Press again to clear 1 finished')` -- the real symptom with the observed value, never a `NoMatches`.

**Files:** `Tests/UI/test_library_shell.py`.
<!-- SECTION:NOTES:END -->
