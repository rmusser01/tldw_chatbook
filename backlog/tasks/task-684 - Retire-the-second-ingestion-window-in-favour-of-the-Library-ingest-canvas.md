---
id: TASK-684
title: Retire the second ingestion window in favour of the Library ingest canvas
status: To Do
assignee: []
created_date: '2026-07-26 03:27'
updated_date: '2026-07-26 04:33'
labels:
  - ingest
  - cleanup
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The app ships two ingestion interfaces. Only the second one's Local Files tab duplicates the Library ingest canvas; its Server Sources, Server Jobs and Web Clipper tabs are server-backed capabilities the canvas has no equivalent for, and the canvas states outright that ingest runs on Local. Retiring the window outright would therefore delete working features, so the three capabilities are ported into the canvas first and the window is deleted last. Tracked as an umbrella over its subtasks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Import sources opens the Library ingest canvas
- [ ] #2 No route or button reaches the retired window
- [ ] #3 Any capability the retired window had that the canvas lacked is available in the canvas
- [ ] #4 The retired window and its now-unused event handlers are deleted
- [ ] #5 The full test suite passes with the window removed
- [ ] #6 Server-backed ingestion is available from the Library ingest canvas,Remote ingest job status is visible alongside local jobs,Web clipping is available from the Library ingest canvas,Import sources opens the Library ingest canvas,No route or button reaches the retired window,The retired window and its now-unused event handlers are deleted,The full test suite passes with the window removed
<!-- AC:END -->
