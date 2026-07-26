---
id: TASK-684.4
title: Delete the retired ingestion window and its dead wiring
status: To Do
assignee: []
created_date: '2026-07-26 04:33'
updated_date: '2026-07-26 04:35'
labels:
  - ingest
  - cleanup
dependencies: []
parent_task_id: TASK-684
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Once every capability has an equivalent in the Library ingest canvas, the second window and everything that exists only to serve it should go, so the two cannot drift apart again.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Import sources opens the Library ingest canvas
- [ ] #2 No route, button or command reaches the retired window
- [ ] #3 The window, its panels and its now-unused event handlers are deleted
- [ ] #4 The full test suite passes with the window removed
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-verify every capability of the retired window has a canvas equivalent (the gate for this task).
2. Point #library-workspace-import-sources at the Library ingest canvas instead of NavigateToScreen('ingest').
3. Remove the 'ingest' ScreenRoute and MediaIngestScreen.
4. Delete MediaIngestWindowRebuilt and its four panels.
5. Delete the handlers that existed only for it -- check local_ingest_events.py and tldw_api_events.py for callers first; both import the window directly today.
6. Delete Tests/UI/test_media_ingest_window_rebuilt.py and any other tests of the removed surface.
7. Full suite plus a live pass over every path that used to reach the window.
<!-- SECTION:PLAN:END -->
