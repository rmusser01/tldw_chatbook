---
id: TASK-21242
title: >-
  Library canvas-sync seam - four correctness gaps left by the TASK-21116
  conversion
status: Done
assignee:
  - rmusser01
created_date: '2026-08-23'
labels:
  - bug
  - library
  - ux
  - regression
dependencies: []
priority: medium
---

## Description

Source: close-out of the 2026-08-22 holistic performance review burn-down; four of the five
Minors from the TASK-21116 review, deliberately not fixed in that round. The fifth (a
remaining whole-screen recompose site) is filed separately as TASK-21243 because it is
performance work, not correctness work.

TASK-21116 continued converting `library_screen.py`'s whole-screen `refresh(recompose=True)`
sites onto the canvas-scoped `library_canvas_sync` seam. Four correctness gaps in the new
seams (line cites are from the review round; the symbols are the durable reference):

1. Builder side effects are not undone when a sync ends SUPERSEDED or FAILED
   (`_pop_library_media_arrival_note`, `_library_media_composed_detail`), so the ingest
   "matched an existing item" notice can be consumed and then never shown to the user.
2. The notes arm ignores `_library_notes_source == "files"`. Latent today because the boundary
   arm fires first — a wrong predicate waiting for the arm ordering to change.
3. In the `sync_kind is None` branch a `follow_up` is computed and then discarded, dropping
   `restore_focus`; focus is not returned after that path.
4. The armed list-entry-focus re-request that `compose_content` performed is not reproduced by
   the new seams.

Note that the harness for this seam's own test file is currently broken (TASK-21232, 4 failed
on dev), so the seam is less covered than it looks.

## Acceptance Criteria

- [x] A SUPERSEDED or FAILED canvas sync leaves the ingest arrival notice still pending, so it is shown on the next successful sync
- [x] The notes arm's source predicate is correct for `files` as well as `database`, independent of which arm fires first
- [x] Focus is restored after the `sync_kind is None` path, matching the pre-conversion behaviour
- [x] List-entry focus is re-requested after a canvas sync, matching what `compose_content` did
- [x] Each of the four has a test that fails against the current behaviour
- [x] TASK-21116's whole-screen recompose site ratchet stays green

## Implementation Plan

1. Premise-verify all four gaps at dev `9f8533686c` (line cites in the
   description were from the review round; the symbols are durable).
2. Gap 1 (side effects): snapshot/restore pair on the media controller
   (`_snapshot/_restore_library_media_reader_side_effects`) wired into
   `swap_library_browse_route` — the one seam that builds the Reader
   (Media work pane) and can then fail/supersede (adopt refuses, or an
   exception propagates to `_replace_library_browse_canvas`'s fallback).
3. Gap 2 (files source): refuse `kind == "notes"` in `_sync_library_canvas`
   while `_notes_state.source == "files"` — before any DOM resolution, no
   whole-screen fallback (mirrors the route-ownership refusal precedent).
4. Gap 3 (`follow_up` discard): `_project_library_canvas_child`'s
   `sync_kind is None` branch runs the composed `follow_up`
   (restore_focus + then) via `call_after_refresh`, not bare `then`.
5. Gap 4 (armed focus re-request): compose the guarded re-request into the
   canvas sync's follow-up queue as a DEFAULT (runs first; explicit intent
   wins) for the list-row kinds.
6. One test per gap, RED-first verified against the pre-fix files via
   checkout swap; ratchet + seam batteries re-run.

## Implementation Notes (2026-10-11, dev `9f8533686c`)

- **Gap 1**: `library_browse_route_swap.swap_library_browse_route` now
  snapshots `(arrival_note, composed_detail)` before the Media work-pane
  build (`_build_library_media_reader` pops the ingest "Matched an
  existing item" note and advances the composed-detail marker) and
  restores both on the `not adopted` return and on any exception
  propagating to `_replace_library_browse_canvas`'s whole-screen fallback
  — so the fallback recompose's reader build re-reads them. Restore is
  conservative: the note re-arms only while the field is still empty (a
  newer note wins); the marker restores unconditionally (the discarded
  build rendered nothing). Helpers:
  `_snapshot/_restore_library_media_reader_side_effects` on
  `LibraryMediaController` (`library_media_controller.py`).
- **Gap 2**: `_sync_library_canvas`'s notes arm
  (`canvas_sync.py`) refuses outright when
  `_notes_state.source == LIBRARY_NOTES_SOURCE_FILES` — before
  `query_one("#library-notes-canvas")`, no whole-screen fallback, mirroring
  the route-ownership refusal: under Files the route's canvas owner is the
  file-notes workspace and there is nothing for a database-notes sync to
  recover. `getattr`-defensive so bare-controller receivers without
  `_notes_state` keep their existing behaviour.
- **Gap 3**: `_project_library_canvas_child`'s `elif then is not None:`
  became `elif follow_up is not None:` running the COMPOSED follow-up —
  previously the branch ran bare `then`, dropping `restore_focus`
  (and discarding the computed restore outright when `then` was None) for
  widgets with no owner-driven sync (loading Static, media viewer,
  collections).
- **Gap 4**: `_sync_library_canvas` queues the guarded continuation
  `_focus_library_list_entry_if_current(generation)` for the list-row
  kinds (`conversations/media/media-trash/notes/prompts/skills`) while
  `_library_pending_list_entry_focus` is armed — as a DEFAULT
  (`queue_default_after_recompose`), so an explicit action intent queued
  by the same sync still wins focus (the composition discipline of
  task-32539). Fire-time `getattr` resolution keeps bare-controller
  receivers safe.
- **Tests**: `Tests/UI/test_library_canvas_sync_gaps_t21242.py` (7 tests).
  RED-first verified by `git checkout HEAD -- <the 4 production files>`
  swap: the five gap tests fail at base
  (`5 failed, 2 passed`), all 7 green after
  (`7 passed in 6.61s`); the two base-green tests are deliberate controls
  (the database-source non-refusal, and the pre-existing
  default-queue-composition discipline pin).
- **No regression**: `Tests/UI/test_library_canvas_scoped_sync.py` +
  `Tests/UI/test_library_per_click_recompose_t21116.py` +
  the new file: `25 passed`. `Tests/UI/test_library_canvas_sync_defects.py`:
  16 failed at base AND after — `diff` of the sorted FAILED lists is empty
  (the pre-existing shared-profile `RecoveryRequired` setup class; CI runs
  these on a real profile). Ratchet
  (`test_library_recompose_ratchet.py::test_library_screen_whole_screen_recompose_count_is_ratcheted`):
  census 67 at base and after — **pre-existing red at dev**
  (67 > pin 63; dev's CI last ran 2026-09-06). This change adds zero
  whole-screen recompose statements; TASK-21243's conversion in the same
  PR brings the census under the pin.
- **Ruff**: changed files carry the same error classes as base (UP037 /
  BLE001 / I001 pre-existing); the one NEW finding my first draft added
  (quoted annotation) was fixed; `ruff format` hunk-count identical to
  base on every touched production file (pre-existing drift untouched).
- **ADR required: no** — correctness repairs inside an existing UI seam;
  no schema, sync-policy, boundary, or contract change.
