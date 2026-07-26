---
id: TASK-715
title: Console Details tray label wraps duplicate Handoff rows and truncated values
status: To Do
assignee: []
created_date: '2026-07-26 17:05'
labels:
  - ux
  - console
  - workspaces
  - copy
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
At rail width the Server handoff label wraps leaving an orphaned lowercase 'handoff' line; two different rows are both labeled Handoff (package list vs ACP status); values truncate ('Off in Default work…', 'ACP handoff: Not co…'); and the tray presents jargon-dense rows (handoff package, ACP task/run package, audit) for features that have no production writer today (server/sync/runtime/ACP states are code-verified unreachable). Finding M4; captures cap-10, cap-18, cap-19.

Source: workspace-settings UX review baseline, Docs/superpowers/qa/workspace-settings-ux-2026-07-26/report.md
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No label in the Details tray wraps mid-phrase at the rail's real width
- [ ] #2 Each Details row has a unique label distinguishing package handoff from ACP handoff
- [ ] #3 Truncated values expose their full text (tooltip or wrap-by-design)
- [ ] #4 Rows whose backing feature cannot be configured anywhere in the UI are hidden or collapsed behind a single not-configured line
<!-- AC:END -->
