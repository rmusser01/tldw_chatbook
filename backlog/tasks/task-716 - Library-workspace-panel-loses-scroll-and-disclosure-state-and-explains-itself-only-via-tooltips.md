---
id: TASK-716
title: Library workspace panel loses scroll and disclosure state and explains itself only via tooltips
status: To Do
assignee: []
created_date: '2026-07-26 17:05'
labels:
  - ux
  - library
  - workspaces
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The workspace management surface lives at the bottom of the Library rail's collapsed Details disclosure. After Create local workspace the rail recomposes: scroll returns to top, the disclosure re-collapses, and the Active row that would confirm the action is hidden. The disabled Use in Console button explains its blocked state only via tooltip, and clicking it gives zero feedback. Finding M5; captures cap-20-25.

Source: workspace-settings UX review baseline, Docs/superpowers/qa/workspace-settings-ux-2026-07-26/report.md
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Disclosure open state and rail scroll position survive a recompose triggered by workspace actions
- [ ] #2 After creating a workspace the updated Active row (or an equivalent confirmation) is visible without re-navigating
- [ ] #3 A disabled Use in Console press surfaces its reason visibly (not tooltip-only)
<!-- AC:END -->
