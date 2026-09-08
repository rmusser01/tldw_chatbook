---
id: TASK-32041
title: >-
  canvas_sync's shared dispatchers take two receiver types with no guard - 8 of
  10 Library controllers have no state accessor
status: To Do
assignee: []
created_date: '2026-09-08 15:04'
labels:
  - library
  - tests
  - tech-debt
  - library-decomposition
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`canvas_sync.py`'s two shared dispatchers (`_apply_library_row_toggle`, `_sync_library_canvas`) are handed a `LibraryScreen` by some callers and a subsystem CONTROLLER by others, and nothing checks that the attribute spellings inside a given kind's leg resolve on both. The flat `_library_<kind>_<field>` spelling resolves on either receiver (a controller's own permanent generated shim loop exposes it); the DOTTED `_<kind>_state.<field>` spelling resolves only on a receiver that declares a `_<kind>_state` accessor property, and 8 of the 10 controllers the decomposition program created do not. Because the dispatchers wrap everything in `except Exception` and fall back to `screen.refresh(recompose=True)`, getting this wrong is silent: no exception, no red test, and the whole-screen recompose the Tier-1 targeted-sync design exists to avoid fires on every press. This has already happened once in production. Wave-7's media retarget dotted `screen._media_state.selected_media_id` in the `_sync_library_canvas` media leg while `LibraryMediaController` -- whose `handle_library_media_select_all`/`handle_library_media_select_clear` call that dispatcher with a bare `self` -- had no `_media_state` property, so every media Select-all/Clear press silently full-screen recomposed. Found by wave-8's receiver census and fixed at the program close. The opposite decision exists too, in the same file: the `search` leg deliberately uses the FLAT name with a comment saying the controller has no `_rag_search_state` at all. Two authors met the same hazard and only one of them left something that would catch the next person. Today's exposure is closed by inspection (only the media and notes legs read a dotted state path, and both controllers now declare the accessor), so this is a guard-and-contract task, not a live bug.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A test derives, from source, which `kind` legs of each dispatcher read a dotted `screen._<x>_state.<field>` path AND which receivers reach that leg, and fails when a reaching receiver cannot resolve the path
- [ ] #2 That guard is mutation-verified in both directions: removing an accessor property reds it, and reverting a dotted spelling to the flat name does not red it spuriously
- [ ] #3 Every controller the decomposition program created either declares its own `_<subsystem>_state` accessor property, or is recorded as deliberately not declaring one with the reason (the `search` leg's existing comment is the model)
- [ ] #4 The receiver census lives in the test rather than in prose, so it re-derives on every run instead of going stale
<!-- AC:END -->
