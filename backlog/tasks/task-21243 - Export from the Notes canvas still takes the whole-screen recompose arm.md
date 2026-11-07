---
id: TASK-21243
title: >-
  Export from the Notes canvas still takes the whole-screen recompose arm
status: Done
assignee:
  - rmusser01
created_date: '2026-08-23'
labels:
  - library
  - performance
dependencies: []
priority: medium
---

## Description

Source: close-out of the 2026-08-22 holistic performance review burn-down; the fifth Minor
from the TASK-21116 review. Split from TASK-21242 (the four correctness gaps) because this is
the same class of work TASK-21116 itself did — moving another site onto the seam.

The holistic review counted **~105 whole-screen `refresh(recompose=True)` sites** in
`library_screen.py` (99 on `self.`), on a screen that grew from 26k to 34.8k lines, and
TASK-21116 converted the confirmed-hot per-click ones to the canvas-scoped seam. Export
initiated from the Notes canvas still routinely takes the structural whole-screen arm
(`library_screen.py:29616` reaching `:8556` at the time of the review) rather than a
canvas-scoped sync — so a routine user action still recomposes the largest screen in the app.

## Acceptance Criteria

- [x] Export initiated from the Notes canvas performs a canvas-scoped sync, not a whole-screen recompose
- [x] A test asserts the recompose count for that action and fails if the whole-screen arm is taken
- [x] Export behaviour and the resulting UI state are unchanged
- [x] The whole-screen recompose site ratchet moves down and stays green

## Implementation Plan

1. Premise-verify at dev `9f8533686c`: the Notes canvas's "Export..." press
   routes through `_open_library_export_canvas` ->
   `_project_library_export_canvas` -> `_apply_library_open_item_surface`,
   whose boundary arm (`strip_mounted != strip_needed or mounted_reader !=
   destination_reader`) always fired for the adaptive->ordinary crossing
   (`.library-notes-route` mounted, destination selector None) and paid
   `await self.recompose()` -- a whole-screen rebuild.
2. Convert the adaptive->ordinary direction of the boundary arm: new
   `leave_library_browse_shell_for_ordinary` in `library_browse_route_swap.py`
   removes the source strip and swaps the shell grid's one adaptive child
   for the ordinary trio, built by a NEW shared constructor
   (`_build_library_ordinary_shell_trio`) that `compose_content` also uses,
   so a leave cannot drift from a compose. Refusals/failures keep the
   structural arm unchanged.
3. Convert the export canvas's own five whole-screen arms (submit, quality
   open/close, quality choice, destination apply/refuse) to
   `_sync_library_canvas(screen, "export", then=...)` with their focus /
   worker-start follow-ups riding the canvas's post-recompose hook.
4. Lower the ratchet pin to the new measured census.

## Implementation Notes (2026-10-11, dev `9f8533686c`)

- **Open path**: `leave_library_browse_shell_for_ordinary`
  (`library_browse_route_swap.py`) guards the grid shape (exactly one
  adaptive child of `#library-shell-grid`), removes the Notes source strip
  via the existing `_apply_library_notes_source_strip`, releases mouse
  capture, swaps the adaptive shell for the ordinary
  handle/rail/`#library-canvas` trio, and mounts the host's always-present
  chrome (emergency return + `#library-canvas-route-content`) inside it.
  The destination child is NOT mounted there: the caller falls through to
  the strict canvas-child replacement, which projects the export canvas
  into the fresh route-content host with the usual generation/route
  guards. The boundary arm now tries this leave only for the
  adaptive->ordinary direction (`destination_reader is None and
  mounted_reader is not None and not strip_needed`) -- the ordinary->
  adaptive arrival keeps the sanctioned structural arm. compose_content's
  ordinary tail and the leave build from the ONE
  `_build_library_ordinary_shell_trio` (+ `_library_single_notes_stage`)
  extraction, so the two composers cannot drift (a first attempt that
  pre-attached the host chrome as constructor children broke compose --
  Textual's compose context cannot re-enter an attached widget; the
  pieces are returned unattached and each path mounts/yields them in its
  own way).
- **Export-canvas arms**: the five `self.refresh(recompose=True)` sites in
  `library_export_controller.py` (submit :937, quality open/close :1291,
  quality choice :1315, destination apply :1409, refuse :1422 at base) now
  call `_sync_library_canvas(self._screen, "export", then=...)`; the
  emergency-guard sync + worker start (submit) and the focus follow-ups
  (quality) run as the canvas's post-recompose follow-up -- the ordering
  guarantee the PostRecomposeCallback hook exists for. A seam failure
  still takes its own whole-screen fallback and re-runs the follow-up
  there.
- **Ratchet**: census at base measured **67 vs the 63 pin -- pre-existing
  red on dev** (the pin was last set 2026-09-01; dev's CI last ran
  2026-09-06, so the drift never bit). This task removes 5 statements
  (the five export arms) -> census **62**; the pin is lowered 63 -> 62
  per the file's own "LOWER this pin when you remove sites" rule.
  `test_library_recompose_ratchet.py` all 7 green including the anti-slack
  guard (slack 0).
- **Tests**: `Tests/UI/test_library_export_open_t21243.py` (2 tests).
  `test_export_open_from_notes_is_canvas_scoped` pins ZERO
  `refresh(recompose=True)` AND ZERO `Screen.recompose()` for the press,
  export canvas mounted pre-scoped to notes, origin row recorded,
  `#library-canvas-route-content` present. Mutation check: forcing the
  leave arm's condition to `False` makes it FAIL (the structural arm is
  caught by the spy); restored, both pass.
- **No regression**: export suites
  (`test_library_crit10_export.py`, `test_library_export_cancel.py`,
  `test_library_media_export_no_freeze.py`) 24 passed;
  `test_library_canvas_scoped_sync.py` +
  `test_library_per_click_recompose_t21116.py` (incl. the
  export-from-media pin) 18 passed; multiselect/phase-c files show the
  SAME 11 failures at base and after (`diff` of sorted FAILED lists
  empty -- the pre-existing shared-profile `RecoveryRequired` class).
- **Ruff**: new findings none (format hunk-counts restored to base on
  every touched production file).
- **ADR required: no** -- routing change inside an existing UI seam with
  an unchanged data model; the canvas-sync seam family is already
  governed by the task-15457/21116 design records.
