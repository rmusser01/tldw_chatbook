---
id: TASK-14800
title: >-
  Library ingest live-update test races the rail recompose it is waiting for
  (NoMatches on #library-row-browse-media)
status: To Do
assignee: []
created_date: '2026-08-09 17:30'
labels:
  - library
  - tests
  - flake
  - dev-baseline
priority: medium
dependencies: []
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
- [ ] #1 `test_library_shell_ingest_canvas_live_updates_without_manual_recompose` survives a loop under deliberate CPU contention without a `NoMatches` failure
- [ ] #2 A failure of that test reports the real symptom (count never incremented, with the last observed label) rather than a `NoMatches` from a transient recompose window
- [ ] #3 Every other poll loop in `Tests/UI/test_library_shell.py` that reads a widget which the awaited change recomposes is either tolerant of the gap or shown not to need it
<!-- AC:END -->
