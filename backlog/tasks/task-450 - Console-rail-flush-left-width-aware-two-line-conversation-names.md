---
id: TASK-450
title: 'Console rail: flush-left width-aware two-line conversation names'
status: In Progress
assignee: []
created_date: '2026-07-21 07:07'
updated_date: '2026-07-21 07:27'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Left-rail conversation names are hard-truncated at 20 chars and indented by a marker prefix. Implement the approved design: flush-left names wrapping to up to 2 width-aware lines, metadata line kept, guarded relabel on width change. Spec: Docs/superpowers/specs/2026-07-20-console-rail-conversation-row-layout-design.md
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Conversation names render flush left with no marker prefix
- [ ] #2 Names wrap to at most 2 lines at the rail's measured width and ellipsize only when 2 lines are insufficient
- [ ] #3 Metadata line renders as the row's final line and is cell-truncated to the row budget
- [ ] #4 Precomputed list heights match rendered row heights for mixed wrapped/badge rows
- [ ] #5 No recompose oscillation when the rail scrollbar toggles or the terminal resizes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-07-20-console-rail-row-wrap.md
<!-- SECTION:PLAN:END -->
