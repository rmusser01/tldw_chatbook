---
id: TASK-15670
title: 'Change review lists one root twice under a single turn entry'
status: To Do
assignee: []
created_date: '2026-08-11 21:30'
labels:
  - console
  - change-review
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`AgentRunsChangeReviewProvider.turns()` groups by run id, and after PR 3a-1 Task 6c a run can hold both a turn row and a post-turn (survivor) row for the same root. Nothing breaks and multi_root labelling still works, but the review screen's file list can show the same path twice with no visible reason. Splitting the selector by `kind` needs a ReviewTurn key that is not the run id, which is a UI change Task 6c deliberately did not fold in.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A run holding both a turn window and a survivor window lists each root once, or labels the two entries so the repetition is explained
- [ ] #2 multi_root labelling still behaves as it does today
- [ ] #3 The provider's existing tests still pass and a new one fails when the duplicate reappears
<!-- AC:END -->
