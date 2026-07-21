---
id: TASK-413
title: Skills editor - visible unsaved-changes feedback instead of silent veto
status: To Do
assignee: []
created_date: '2026-07-21 15:18'
labels:
  - skills
  - ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
P0-adjacent from the 2026-07-21 Skills UX/NNG review (verified live). Where the dirty-edit guard does fire (Back to list button and skill-row-to-row switches) it is a silent veto: the click does nothing - no toast, no status change, no prompt. The Back button appears broken. Combined with task-412 the surface is incoherent: some exits silently block, others silently discard. One consistent pattern is needed across all skill-editor exits. NNG heuristic 1 (visibility of system status).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Attempting to leave the skill editor with unsaved changes produces visible feedback explaining why navigation was blocked,User has an explicit path to leave without saving (discard affordance or prompt) as well as to save,The same pattern applies to Back to list and skill-row switches and (with task-412) rail-row switches,Behavior covered by tests
<!-- AC:END -->
