---
id: TASK-31950
title: >-
  Library media - four post-recompose focus and sync callbacks still run through
  screen call_after_refresh
status: To Do
assignee: []
created_date: '2026-09-07 08:25'
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
- [ ] #1 After Escape closes the Reader's More menu, focus is on a mounted widget and the next key is delivered
- [ ] #2 All four sibling sites schedule through the viewer's queue_after_recompose or the _after_library_media_viewer_sync seam instead of a screen-level call_after_refresh
- [ ] #3 The pins wait for the mounted widget and assert the focused widget is attached, so a focused orphan fails them
<!-- AC:END -->
