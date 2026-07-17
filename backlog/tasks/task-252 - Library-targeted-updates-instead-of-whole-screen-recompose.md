---
id: TASK-252
title: Library: targeted sync_state updates instead of 124 whole-screen recomposes
status: To Do
assignee: []
created_date: '2026-07-16 14:30'
labels: [performance, library]
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
library_screen.py calls self.refresh(recompose=True) — full remove/remount of nav, footer, ~20-row rail, and 50-100-row canvas — from 124 sites including per-row checkbox handlers. LibraryRail/LibraryMediaCanvas/LibraryConversationsCanvas.sync_state() exist for targeted updates and have ZERO callers. This exact pattern caused the app-wide mouse-capture bug base_app_screen.py works around. Stage by interaction class (checkbox toggles first). Full analysis with measurements: Docs/Design/2026-07-16-performance-audit.md (§P1 B2).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Per-row selection/checkbox interactions no longer recompose the screen (targeted row/canvas updates; verified by interaction tests)
- [ ] #2 Rail counts stay consistent with canvas state across the converted interactions
- [ ] #3 Remaining recompose sites inventoried with a justification or follow-up
- [ ] #4 Live QA on the converted flows
<!-- AC:END -->
