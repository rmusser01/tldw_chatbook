---
id: TASK-289
title: Media search imports deleted UI.MediaWindow module — always errors
status: To Do
assignee: []
created_date: '2026-07-17 21:30'
labels: [bug, media, library]
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during task-283 (PR review sweep of the debounced-search paths): Event_Handlers/media_events.py's perform_media_search_and_display imports UI.MediaWindow, a module deleted by the Library redesign — the import raises, so media search always renders its "Error loading…" state on current dev. The search itself (search_media_db) works; only the display path is broken. Fix: route the display through the current Library media surface (or the appropriate current widget), and add a regression test that exercises the real handler end-to-end (the deleted-module import would have been caught by any unmocked call).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Media search displays results through a module that exists; the handler imports cleanly
- [ ] #2 An unmocked test drives perform_media_search_and_display end-to-end (import failure class pinned)
<!-- AC:END -->
