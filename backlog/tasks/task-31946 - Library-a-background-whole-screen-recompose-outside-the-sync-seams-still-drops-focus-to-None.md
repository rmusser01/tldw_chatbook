---
id: TASK-31946
title: >-
  Library - a background whole-screen recompose outside the sync seams still
  drops focus to None
status: Done
assignee: []
created_date: '2026-09-07 08:25'
updated_date: '2026-09-07 20:03'
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
- [x] #1 A bare refresh(recompose=True) on a Library screen leaves focus on the equivalent widget or a defined fallback, never None
- [x] #2 The restore lives at one shared seam (BaseAppScreen), not at each call site
- [x] #3 A pin drives a background recompose and asserts the focused widget is mounted
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Pin: a bare `screen.refresh(recompose=True)` with focus on a widget (Media route and a non-Media route) ends with `screen.focused` a mounted widget — the same-id widget when it still exists, else the defined fallback — never None. 2. Put the restore at ONE shared seam and remove the per-site copies.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`BaseAppScreen.refresh` captures the focused widget's identity and queues exactly one restore per whole-screen recompose through an overridable `restore_focus_after_recompose`; `LibraryScreen` overrides it with the Media rules (PR F's capture/restore) and then the base tail, and PR F's own `call_after_refresh` restore is gone, so a double restore is impossible by construction. Guards: restore only when an id-bearing widget actually held focus; stand down while a Library one-shot focus channel (the armed list-entry focus, the Find token) can land the focus itself. Fallback when the captured widget vanished: the first focusable inside the screen's content area (never the nav bar, where a blind Enter leaves the screen), then the focus chain. Two InvokeLater hops keep the restore behind a subclass's own post-recompose passes (Widget._on_hide blurs, so an earlier restore lands at None). Two pins drive background recomposes on a Media and a non-Media route. Not live-provokable; the pins are the evidence.
<!-- SECTION:NOTES:END -->
