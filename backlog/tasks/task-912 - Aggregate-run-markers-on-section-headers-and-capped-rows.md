---
id: TASK-912
title: 'Aggregate run markers on section headers and capped rows'
status: To Do
assignee: []
created_date: '2026-07-27 03:55'
labels: [console, fleet-ux]
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two fleet-marker visibility gaps from the parallel-agents train: (1) top-level conversation-browser sections (Starred/Workspaces/Chats) have no run_marker aggregate, so collapsing a whole section hides every marker beneath it; (2) an expanded workspace group with more than the 12-row cap can push a marked row past the cap with no marker surfaced (header aggregate only renders when collapsed). Collapsed workspace GROUP headers already aggregate the most-urgent glyph.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Collapsed top-level sections surface the most-urgent marker among their contents.
- [ ] #2 A marked row beyond the group row cap surfaces its marker (e.g. header aggregate also when expanded-but-capped, or overflow row indicator).
- [ ] #3 Urgency order matches the existing group-header aggregation.
<!-- AC:END -->
