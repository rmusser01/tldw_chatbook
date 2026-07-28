---
id: TASK-1232
title: 'Fleet capability discoverability: Help, coach-mark, footer'
status: To Do
assignee: []
created_date: '2026-07-28 09:30'
labels: [console, ux, docs, uat]
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expert UAT F2: nothing at rest communicates that each Console tab runs its own agent in parallel under a cap. F1 Help covers panes/transcript/composer only (zero mentions of agents, approvals, workspaces, parallel runs); the footer omits Alt+W and Alt+1..9; the capability teaches itself only after accidental use.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 F1 Help gains an Agents section (tabs=agents, cap + where to change it, approval flow, marker legend, Alt+W / Alt+1..9).
- [ ] #2 A one-time dismissible coach-mark on first second-tab creation states the parallel model and the cap.
- [ ] #3 Footer (or Help) lists the workspace/tab-jump hotkeys.
<!-- AC:END -->
