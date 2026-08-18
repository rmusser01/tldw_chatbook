---
id: TASK-17654
title: 'Console: raise the composer draft cap from 4 to 8 rows'
status: To Do
assignee: []
created_date: '2026-08-17'
labels:
  - console
  - ux
dependencies:
  - task-17651
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Owner follow-up from the 2026-08-17 bottom-stack decisions: after the composer flattens to 1-4 total rows (TASK-17651), raise the visible draft capacity to 8 rows for long prompts. This is deliberately separate because three things must move together: `MAX_DRAFT_ROWS`, the draft Static's `max-height`, and the viewport-window slicing logic (`_wrap_draft_line_slices`) that keeps the caret visible — changing only the constant would break the draft windowing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 The draft area can display up to 8 rows and the composer's total height never exceeds 8 rows plus its (zero) chrome
- [ ] #2 Draft windowing and caret visibility behave correctly at every draft height up to the new cap, including paste-collapse and ghost text
- [ ] #3 Geometry pins are updated, and composer growth never starves the transcript below usable height at supported terminal sizes (the compact-mode threshold class)
<!-- AC:END -->
