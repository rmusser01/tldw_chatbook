---
id: TASK-2033
title: 'Console boot lands on Default workspace even when another is registry-active'
status: To Do
assignee: []
created_date: '2026-08-03 00:45'
labels:
  - console
  - workspaces
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in TASK-1980 live UAT. Settings → Workspaces "Set active" marks a
workspace active in the registry (list shows "UAT Review (active)"), but
after an app restart the Console workspace switcher shows "Default
(everyday chats) (current)" — the Console context does not start on the
registry-active workspace, and the startup session is created tool-less
under Default. May be deliberate ("Switching changes Console context only")
but it surprises: the one thing "Set active" visibly promises is where you
land next. Owner decision wanted: either boot Console on the registry-active
workspace, or rename/re-copy the Settings affordance so it stops implying
that.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Decision recorded: Console boot honors registry-active workspace, or the Settings "Set active" copy/behavior is changed to match reality
- [ ] #2 The chosen behavior is implemented and tested
<!-- AC:END -->
