---
id: TASK-714
title: Workspaces cannot be renamed archived or deleted and names are forced Workspace N
status: To Do
assignee: []
created_date: '2026-07-26 17:05'
labels:
  - ux
  - workspaces
  - feature
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Both create paths force sequential names (Workspace 1, 2, ...) and no surface offers rename, archive, or delete. Combined with silent creation (task-713) and an invisible create button (task-712), real usage accumulates indistinguishable and accidental workspaces with no way to clean up. The retired create-local-copy control id survives as a dim WIP static. Finding M3.

Source: workspace-settings UX review baseline, Docs/superpowers/qa/workspace-settings-ux-2026-07-26/report.md
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A workspace can be given a user-chosen name at or shortly after creation
- [ ] #2 An existing workspace can be renamed from at least one surface
- [ ] #3 A non-default workspace can be archived or deleted with appropriate guardrails for its conversations
- [ ] #4 The Default workspace is protected from destructive lifecycle actions
<!-- AC:END -->
