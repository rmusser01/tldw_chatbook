---
id: TASK-720
title: Storage shows workspaces DB template path and resolved path as conflicting truths
status: To Do
assignee: []
created_date: '2026-07-26 17:05'
labels:
  - ux
  - settings
  - storage
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Storage category's Workspaces input displays the default template path while the caption below shows the actually-resolved per-profile path - two different paths for a restart-gated, data-loss-adjacent setting (cap-27). Users cannot tell which one is live. Finding m4.

Source: workspace-settings UX review baseline, Docs/superpowers/qa/workspace-settings-ux-2026-07-26/report.md
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The input and its caption cannot be read as two different current locations
- [ ] #2 The actually-active DB path is unambiguous in the Storage category
<!-- AC:END -->
