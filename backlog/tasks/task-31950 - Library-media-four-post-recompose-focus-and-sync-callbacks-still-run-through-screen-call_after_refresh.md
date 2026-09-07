---
id: TASK-31950
title: >-
  Library media - four post-recompose focus and sync callbacks still run through
  screen call_after_refresh
status: Done
assignee: []
created_date: '2026-09-07 08:25'
updated_date: '2026-09-07 20:03'
labels:
  - library
  - media-ux
  - bug
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
H Task 3 review M2, H final review and H2 review (2026-09-06): PR H Task 3 measured that screen.call_after_refresh(<focus ...>) after _sync_library_media_viewer_or_recompose() can focus a button the same recompose detaches, leaving focus on an orphan that swallows every key, Escape included. Four siblings still carry the pattern - library_screen.py ~31992/32001 (Escape closes More; this is the keyboard path, where every key is dead until the user clicks), ~32061, ~34821, and the pair inside _sync_library_media_viewer_state itself (~34602-34607: the mutation-gate sync and the progress restore, harmless today only because both no-op on NoMatches). Their only coverage is a call-recording fake (test_library_media_reader_flow.py ~1218-1222) that asserts the focused id, which an orphaned widget satisfies.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After Escape closes the Reader's More menu, focus is on a mounted widget and the next key is delivered
- [x] #2 All four sibling sites schedule through the viewer's queue_after_recompose or the _after_library_media_viewer_sync seam instead of a screen-level call_after_refresh
- [x] #3 The pins wait for the mounted widget and assert the focused widget is attached, so a focused orphan fails them
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Census: every `_sync_library_media_viewer_or_recompose()` / `viewer.refresh(recompose=True)` followed within ~14 lines by `call_after_refresh(`. 2. Route the remaining sites through the viewer-scoped seam; make the census a test. 3. Mounted-focus pins (identity + is_attached + a following key).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Census on the merge-base found two remaining sites, both in `_sync_library_media_viewer_state`'s tail (the mutation-gate sync and the progress restore right after its own `viewer.refresh(recompose=True)`); the other siblings the task named had already landed with PR H2. The seam's ordering half is now `_queue_after_library_media_viewer_recompose(callback, viewer=None)` (same mechanism: `_recompose_required` gate, `call_after_refresh` fallback, chained ahead of the pending restore under `finally`); `_after_library_media_viewer_sync` delegates to it and the tail passes the viewer it holds. Census-as-test pins zero sites; the gate pin asserts the recomposed Save is attached before reading its disabled state. Live: Escape from inside More, Find open/close and Read → Analysis → Read all keep keys working. Riders: a whole-screen prune between queueing and the viewer's recompose drops the queued gate/restore (the exposure PR H2 accepted; both are query-guarded no-ops); the gate is chained ahead of the focus restore.
<!-- SECTION:NOTES:END -->
