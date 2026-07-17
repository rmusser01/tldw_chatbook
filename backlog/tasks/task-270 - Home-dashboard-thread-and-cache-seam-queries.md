---
id: TASK-270
title: Home: thread + cache dashboard seam queries; targeted rail/canvas updates
status: To Do
assignee: []
created_date: '2026-07-16 14:30'
labels: [performance, home]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
home_screen._build_dashboard_input runs 3 synchronous DB/repository queries (watchlist snapshot, notification queue limit=100, server-event feed) on the UI thread at every compose, triage sync, and rail click with no cross-visit cache — while the sibling _home_content_seam_call already uses asyncio.to_thread correctly. HomeRail/HomeCanvas.sync_state also always recompose. Full analysis with measurements: Docs/Design/2026-07-16-performance-audit.md (§P1 B3).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Dashboard seam queries run off the event loop with a short-TTL cache; Home still reflects fresh data per its seam contract
- [ ] #2 Selection/count-only changes patch targeted widgets instead of recomposing rail/canvas
- [ ] #3 Existing Home triage tests green
<!-- AC:END -->
