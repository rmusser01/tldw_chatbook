---
id: TASK-1563
title: 'Settings: collapse view-only stub categories; fix Follow-up: doubling'
status: To Do
assignee: []
created_date: '2026-07-31 02:00'
labels: [settings, ux, P2]
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique finding (P2): seven categories (Artifacts, Skills, Schedules,
Watchlists, Workflows, MCP Defaults, ACP Defaults) are full navigation peers
whose entire page says "change this elsewhere" -- a third of the rail is
non-actionable prose. Each also renders the shipped copy bug "Follow-up:
Follow-up: ..." (root cause: `_detail_row("Follow-up", contract.follow_up)`
at settings_screen.py:10433 re-prefixes strings already prefixed at
540-665).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 "Follow-up:" renders exactly once per stub page.
- [ ] #2 View-only categories are visually distinguished in the rail (dimmed and/or badged "view") or collapsed under one "Owned elsewhere" group.
- [ ] #3 Rail remains navigable to their detail pages.
<!-- AC:END -->
