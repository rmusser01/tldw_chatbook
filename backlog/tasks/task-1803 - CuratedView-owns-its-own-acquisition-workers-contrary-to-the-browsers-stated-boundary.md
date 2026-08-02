---
id: TASK-1803
title: >-
  CuratedView owns its own acquisition workers, contrary to the browser's stated
  boundary
status: To Do
assignee: []
created_date: '2026-08-02 00:43'
labels:
  - models
  - architecture
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The TASK-596 design states that widgets and views never call preflight/provision/activate/delete -- the host screen owns the worker, so a long download survives its consent dialog being dismissed and cannot be orphaned by a recompose. dev's CuratedView (tldw_chatbook/UI/Screens/model_curated_view.py) carries three @work-decorated methods and drives acquisition itself. This surfaced concretely during the TASK-596 delta port: after a screen-level recompose the orphaned view instance's progress messages were silently dropped, which needed compensating delivery logic to work around. InstalledView and LibraryScreen follow the screen-owns-worker shape; CuratedView is the outlier.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 CuratedView posts intents; LLMScreen owns the preflight/provision workers, matching LibraryScreen
- [ ] #2 A screen-level recompose mid-install cannot orphan the worker or drop progress, without compensating delivery logic
- [ ] #3 The compensating logic added by the delta port is removed once ownership moves
<!-- AC:END -->
