---
id: TASK-631
title: Guard mouse capture across non screen recompose widgets
status: To Do
assignee: []
created_date: '2026-07-25 18:00'
labels:
  - followup
  - uat
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-627 fixed the capture leak for BaseAppScreen recompose, but non-screen widgets that recompose get neither guard (same bug class, different trigger): mcp_rail.py:205, ResultsDashboardWindow, Chatbooks_Window_Improved, Mindmap_Viewer_Window:354, app.py:6024. A capture held by a descendant of any of these at recompose time leaks app-wide.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All non-screen recompose sites release/sweep stale mouse capture like BaseAppScreen
- [ ] #2 Regression test covers at least one non-screen site
- [ ] #3 No legitimate (still-attached) capture is released
<!-- AC:END -->
