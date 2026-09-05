---
id: TASK-31570
title: Library media - F6 focus ring should not start on the two text inputs
status: To Do
assignee: []
created_date: '2026-09-05 03:23'
labels:
  - library
  - media-ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
F6's first two stops are the search input and the Find input, so the keyboard user lands in text boxes before reaching the Items list or the Reader (wave 4 PR B deferred).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 F6 visits the Items row and the Reader content before any text input
- [ ] #2 Both inputs stay reachable in the ring
- [ ] #3 A test pins the order at 235x52
<!-- AC:END -->
