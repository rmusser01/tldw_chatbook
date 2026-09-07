---
id: TASK-31946
title: >-
  Library - a background whole-screen recompose outside the sync seams still
  drops focus to None
status: To Do
assignee: []
created_date: '2026-09-07 08:25'
labels:
  - library
  - media-ux
  - bug
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR F Task 3 residual (2026-09-05): PR F restored focus across the recomposes routed through _sync_library_canvas and the media viewer sync, but any other path calling refresh(recompose=True) on the screen (a background job tick, an ad-hoc repaint) still leaves screen.focused None, so the keyboard is dead until the user clicks. The durable fix is a focus-preserving hook on BaseAppScreen.refresh rather than another per-call-site patch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A bare refresh(recompose=True) on a Library screen leaves focus on the equivalent widget or a defined fallback, never None
- [ ] #2 The restore lives at one shared seam (BaseAppScreen), not at each call site
- [ ] #3 A pin drives a background recompose and asserts the focused widget is mounted
<!-- AC:END -->
