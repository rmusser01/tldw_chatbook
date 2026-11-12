---
id: TASK-35900
title: >-
  Console: a burst on the Assistant or Scope segment repeats its read and its
  no-cards notice
status: To Do
assignee: []
created_date: '2026-10-21 01:23'
labels:
  - console
  - performance
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-34720 made the character picker and the retrieval-scope picker open one dialog per burst: each checks, immediately before its push, that the Console is still the top screen. That check runs after the opener's off-thread read, and the segment handlers are awaited one after another, so a burst of N activations still performs N reads, N-1 of them for a dialog that is then refused. With no saved character cards no dialog opens at all, the Console stays uncovered, and every activation in the burst raises its own 'No characters saved yet' notice. Neither crashes; both are wasted work and noise in exactly the burst TASK-34720 addressed (found by the TASK-34720 review).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A burst of Assistant or Scope activations that opens a dialog performs the character-list or scope read once, not once per activation
- [ ] #2 A burst of Assistant activations with no saved characters shows the 'No characters saved yet' notice once
- [ ] #3 The check immediately before each push stays, so a dialog that appears during the read still blocks a second one
- [ ] #4 Pilot tests on the real ChatScreen count the reads and the notices and fail on the code before the change
<!-- AC:END -->
