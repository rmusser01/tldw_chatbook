---
id: TASK-32339
title: >-
  Review changes action renders under a proper heading
status: To Do
assignee:
  - '@zcode'
created_date: '2026-09-10 12:00'
labels:
  - console
  - ux
  - rail-ux-review
priority: low
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UX review D3. The 'Changes' entry exists in _ACTION_GROUPS but has no matching group in _ROW_GROUPS, so the review-changes button falls into the ungrouped actions loop (console_run_inspector.py ~131-135, ~428-431). Add the heading or re-home the action so it renders grouped like its siblings.

Filed from the 2026-09-10 Console rail UX review (review item D3).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The review-changes action renders inside a labeled group consistent with the run inspector's grouping
- [ ] #2 No other action's grouping changes
- [ ] #3 Widget test asserts the button renders under the expected heading
<!-- AC:END -->
