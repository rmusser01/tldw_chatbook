---
id: TASK-21242
title: >-
  Library canvas-sync seam - four correctness gaps left by the TASK-21116
  conversion
status: To Do
assignee: []
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

- [ ] A SUPERSEDED or FAILED canvas sync leaves the ingest arrival notice still pending, so it is shown on the next successful sync
- [ ] The notes arm's source predicate is correct for `files` as well as `database`, independent of which arm fires first
- [ ] Focus is restored after the `sync_kind is None` path, matching the pre-conversion behaviour
- [ ] List-entry focus is re-requested after a canvas sync, matching what `compose_content` did
- [ ] Each of the four has a test that fails against the current behaviour
- [ ] TASK-21116's whole-screen recompose site ratchet stays green
