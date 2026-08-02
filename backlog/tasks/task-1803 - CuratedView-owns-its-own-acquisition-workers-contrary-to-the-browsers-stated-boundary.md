---
id: TASK-1803
title: >-
  CuratedView owns its own acquisition workers, contrary to the browser's stated
  boundary
status: Done
assignee: []
created_date: '2026-08-02 00:43'
updated_date: '2026-08-02 20:21'
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Merged as PR #1210 on 2026-08-02. LLMScreen now owns the curated preflight/provision workers; CuratedView posts intents only. The PR-1185 delivery fallback chain was DELETED outright (AC 3). Review found and fixed a Critical stale-window regression (self.llm_window points at a closed widget during LabScreen.recompose()'s teardown-to-remount gap; post_message returns False and the tick vanished) plus two further bugs: an except-path that could raise a second exception and strand install state, and the Screen-level fallback skipping LLMManagementWindow's InstalledView mirroring. The last is now closed by hydration on every remount rather than by message replay. 627 passed. Follow-up: TASK-1914 (RemoteView has the same ownership defect with no mitigation).
<!-- SECTION:NOTES:END -->
