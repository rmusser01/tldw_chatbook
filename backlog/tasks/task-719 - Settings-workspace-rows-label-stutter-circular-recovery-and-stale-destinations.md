---
id: TASK-719
title: Settings workspace rows label stutter circular recovery copy and stale destination references
status: To Do
assignee: []
created_date: '2026-07-26 17:05'
labels:
  - ux
  - settings
  - copy
  - workspaces
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Overview rows read like debug output ('Workspace default: Workspace: Workspace 3 (workspace-local-3); ...', 'Sync safety: Collections: Sync: dry-run only; ...'). The recovery copy says to open 'the matching Settings category or destination' without ever naming which destination owns workspace management. Code fallback copy still points at 'Library > Workspaces', a mode that no longer exists (it is now a disclosure under the Library rail's Details section). Findings m1/m2; captures cap-11, cap-27.

Source: workspace-settings UX review baseline, Docs/superpowers/qa/workspace-settings-ux-2026-07-26/report.md
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Workspace and sync rows read as label: value without repeated label fragments
- [ ] #2 Settings copy names the concrete surfaces that own workspace switching and management
- [ ] #3 No user-facing copy references the retired Library > Workspaces mode
<!-- AC:END -->
