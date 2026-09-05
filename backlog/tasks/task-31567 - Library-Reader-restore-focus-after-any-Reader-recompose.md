---
id: TASK-31567
title: Library Reader - restore focus after any Reader recompose
status: Done
assignee:
  - '@claude'
created_date: '2026-09-05 03:22'
updated_date: '2026-09-05 15:34'
labels:
  - library
  - media-ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After any recompose of the adaptive Reader shell, focus falls through to the pane grip. This is why Space collapsed the pane in wave 4 PR B Task 1 and why the retired grip end-caps kept reappearing; every fix so far patched one caller. A general restore-focus-after-recompose seam is needed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After a Reader recompose, focus returns to the widget that held it (row, content, Find input) rather than a pane grip
- [x] #2 Space on a focused Items row never collapses a pane
- [x] #3 Painted tests at 235x52 and 100x30 cover the row, content and Find cases
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Capture the focused identity before a media recompose, restore after; explicit handler targets win; never a grip. 2. Wire at the media branch of _sync_library_canvas and the viewer's PostRecomposeCallback. 3. Painted tests both sizes; live ladder.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
_capture_library_media_focus_identity() + _restore_library_media_focus(previous) at two choke points covering 20 recompose sites; explicit targets (PR E focus_identity, the armed list-entry focus, the Find token) win — detected directly for the two one-shot channels and as "focus already on a non-grip" otherwise; a vanished identity falls back to the list entry. The live pass caught a regression the suite could not (queue_after_recompose replaces, clobbering PR E's queued Undo follow-up) — guarded and pinned. Residual: a bare background refresh(recompose=True) outside the seams still leaves focus None (needs a BaseAppScreen.refresh hook; rider). 8 painted tests (4 × 2 sizes) + 2 clobber pins. Files: library_screen.py, library_media_viewer.py, tests in test_library_media_reader_flow.py / test_library_adaptive_reader_shell.py / test_library_multiselect_media.py.
<!-- SECTION:NOTES:END -->
