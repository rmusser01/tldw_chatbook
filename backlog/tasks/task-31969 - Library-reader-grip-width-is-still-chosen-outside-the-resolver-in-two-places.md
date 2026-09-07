---
id: TASK-31969
title: 'Library reader: grip width is still chosen outside the resolver in two places'
status: To Do
assignee: []
created_date: '2026-09-07 20:27'
labels:
  - library
  - reader
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR N (tasks 31951-31953) made `AdaptiveReaderEffectiveLayout.grip_width` the one source for pane-grip width and re-applies it in `sync_layout`. Two places still pick a width on their own: File Notes' pre-resolve placeholder layout hard-codes the default grip width (a shared import would be circular), and the legacy `LibraryMediaPaneGrip` in `library_media_reader_shell.py` reads `MEDIA_READER_LAYOUT_PROFILE.grip_width` directly (only a test constructs it). A future grip-width change misses both.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every pane-grip width on the Library reader surfaces is read from a resolved layout or one shared constant; a census test pins zero other sites
- [ ] #2 The legacy `LibraryMediaPaneGrip` is retired or routed through the resolver
<!-- AC:END -->
