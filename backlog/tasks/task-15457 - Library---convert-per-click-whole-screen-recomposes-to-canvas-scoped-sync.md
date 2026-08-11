---
id: TASK-15457
title: Library: convert per-click whole-screen recomposes to canvas-scoped sync
status: To Do
assignee: []
created_date: '2026-08-11 12:05'
labels:
  - perf
  - library
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
From the audit: `UI/Screens/library_screen.py` carries 147 statement-level `refresh(recompose=True)` / `await recompose()` sites — regrown past the July task-281 fix (124 sites then). Per-click confirmed: notes row click (`:21914`), select toggle/all/clear (`:21926-:21967`), notes sort/filter strip (`:13236-:13287`), skills row (`:14087`), prompts row (`:16638`), media-type chooser (`:12328/:12358`), ingest checkbox toggles (`:20384`), RAG mode/scope toggles (`:24316/:24342`), choice-strip close (`:18160`), notes-sync toggles (`:19582-:19626`) — each tearing down and remounting the whole 26k-line screen (~120-200 widgets + CSS apply + relayout). Cascades stack: "back to list" = 3 recomposes (documented at `:7580-7597`); note creation = 6 refresh sites; a one-line import status change is a full recompose (`:13732/:16347`). The blocker for the Notes canvas is documented at `:21925`: a `LibraryNotesCanvas` constructor param shadows the `sync_state` hook name, so it was deliberately left un-converted in task-252.

Fix direction: rename the shadowing param to unblock the targeted hook; extend the existing `_sync_library_canvas` (`:1281`) pattern; convert per-click sites to canvas-scoped sync, keeping whole-screen recompose only for true canvas transitions. Stability constraints: the recompose-lifecycle traps from the July programme apply (fresh widgets carry current state — per-widget caches must live on the widget instance, not screen dicts); convert in reviewed slices with the existing recompose-discipline patchers (`_apply_library_row_toggle` `:1188` is the model), each slice behavior-pinned before conversion. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 None of the confirmed per-click interactions (notes row/toggles, sort/filter strips, media-type chooser, ingest toggles, RAG toggles, status lines) triggers a whole-screen recompose (evidence per site class)
- [ ] #2 The LibraryNotesCanvas targeted hook is unblocked and used; behavior of every converted surface unchanged (tests)
- [ ] #3 Remaining recompose site count re-measured and recorded; per-click latency before/after on the notes canvas
<!-- AC:END -->
