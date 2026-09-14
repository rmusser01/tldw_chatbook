---
id: TASK-32578
title: >-
  Library Notes: three lasting-sync refusal reason codes reach the row with no
  copy
status: To Do
assignee: []
created_date: '2026-09-14 22:46'
labels:
  - library
  - notes
  - rider
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Declined deliberately by wave-4 group 3 (task-32534) rather than guessed at: three lasting-sync reason codes have no entry in the refusal copy table, so a root that hits one falls through to its category line instead of naming the cause. Writing copy for them inside that task would have meant inventing text for refusals nobody has reproduced, which is how a false guide sentence gets written. The right shape is to reproduce each refusal first, then write copy that names the cause and the next action, in the same grammar as 'Check failed — folder is paused · Next: Resume'.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each of the three uncovered reason codes is reproduced, by test or by a driven route, before copy is written for it
- [ ] #2 Each gets a reason clause and a next action naming a control that is on screen
- [ ] #3 The reason is never the exception text, matching task-32534's contract
- [ ] #4 The refusal copy table is pinned so a future code added without copy fails a test rather than shipping silently
<!-- AC:END -->
