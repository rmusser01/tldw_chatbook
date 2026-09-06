---
id: TASK-31797
title: Library items pane empty ('0 of 0 - Total unavailable') after 'Open in Library' deep-link from the import queue
status: To Do
assignee: []
created_date: '2026-09-05 19:15'
labels:
  - bug
  - ui
  - library
  - ingest
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the 2026-09-05 pre-release live UAT sweep (fresh scratch profile, dev tip 8e9d1128d4, real tmux-driven app). Import a local .md via the Library import flow, wait for the job's done state, click 'Open in Library': the reader opens the item correctly but the middle Items pane shows '0 of 0 - type: None' / 'No page loaded - Total unavailable / Page boundary is unknown' and never recovers on its own. Clicking 'Media (1)' in the left rail populates it. A user arriving via the deep-link sees an apparently empty library beside their open item.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The 'Open in Library' deep-link lands with the items list populated (at minimum the opened item's page loaded).
- [ ] #2 Regression test for the deep-link path asserting a non-empty items page.
<!-- AC:END -->
