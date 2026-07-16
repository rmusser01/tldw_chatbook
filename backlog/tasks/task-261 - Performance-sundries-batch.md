---
id: TASK-261
title: Performance sundries: bg-effect frame cache, token-count gate, SELECT-1 ping, picker ctor I/O, MCP rebuild diffing, browser indexes
status: To Do
assignee: []
created_date: '2026-07-16 14:30'
labels: [performance, ui]
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Bounded small fixes from the audit: console_background_effect render_line recomputes the full W×H grid PER LINE = O(W·H²)/repaint (cache per frame tick); the 10s footer token-count re-tokenizes the whole visible history without a dirty check (app.py:5950-5954); get_connection pings SELECT 1 per query (~2× raw call count); BookmarksManager.__init__ does 5 sync Path.exists() (cloud-dir stalls) + first-run TOML write on every picker construction; MCP rail/servers-table/hidden-Tools-canvas full rebuilds ×2 per lifecycle action (N+2 store loads already tracked in task-236); consider indexes on conversations.deleted/last_modified for the browser ORDER BY at scale. Full analysis with measurements: Docs/Design/2026-07-16-performance-audit.md (§P3 D3).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each listed item fixed or explicitly declined with reasoning in the task notes
- [ ] #2 No behavior changes (existing suites green)
<!-- AC:END -->
