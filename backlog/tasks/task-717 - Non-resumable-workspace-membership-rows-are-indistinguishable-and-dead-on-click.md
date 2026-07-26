---
id: TASK-717
title: Non-resumable workspace membership rows are indistinguishable and dead on click
status: To Do
assignee: []
created_date: '2026-07-26 17:05'
labels:
  - ux
  - console
  - workspaces
  - bug
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Membership rows that cannot be resumed (role source, or conversation records missing from the chat DB) render identically to openable conversation rows and produce no reaction at all when clicked - no toast, no navigation. Live-verified with a ghost membership (cap-26). Finding M6.

Source: workspace-settings UX review baseline, Docs/superpowers/qa/workspace-settings-ux-2026-07-26/report.md
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Rows that cannot be opened are visually distinct from openable conversation rows
- [ ] #2 Clicking a non-resumable row produces visible feedback explaining why nothing opened
- [ ] #3 A membership whose conversation record is missing surfaces a recovery hint that matches an affordance that actually exists
<!-- AC:END -->
