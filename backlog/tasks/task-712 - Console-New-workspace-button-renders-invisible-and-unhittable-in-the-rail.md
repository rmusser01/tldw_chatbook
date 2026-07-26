---
id: TASK-712
title: Console New-workspace button renders invisible and unhittable in the rail
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
The Session action row's 12-column left margin plus two 16-column-minimum buttons overflow the ~37-column Console rail, so the New button's label renders entirely outside the clip while a blank strip stays clickable (live-verified: sweep-clicking blank space created a workspace). The only Console affordance for creating a workspace is invisible, while the adjacent copy tells users to add another workspace. The comment at console_workspace_context.py:751-766 documents this exact overflow failure mode for a third button; the margin re-broke the original pair. Finding C1.

Source: workspace-settings UX review baseline, Docs/superpowers/qa/workspace-settings-ux-2026-07-26/report.md
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both Switch and New are fully visible and clickable at the rail's real width
- [ ] #2 A regression test asserts both action buttons' regions fit within the rail clip
- [ ] #3 No invisible clickable region remains in the Session action row
<!-- AC:END -->
