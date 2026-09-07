---
id: TASK-31945
title: >-
  Library sibling canvases - row buttons drop clicks during the 0.2 s active
  flash
status: To Do
assignee: []
created_date: '2026-09-07 08:25'
labels:
  - library
  - media-ux
  - bug
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR F Task 2: a Textual Button swallows a click that lands inside the previous press's 0.2 s active-effect flash. The media row buttons were fixed with active_effect_duration = 0; the conversations, notes and prompts row buttons on the sibling library canvases still carry the default and drop fast successive clicks the same way.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A second click on a sibling-canvas row button within 0.2 s of the first is delivered
- [ ] #2 All four library canvases share one row-button press behaviour
- [ ] #3 At least one sibling canvas has a pin for the fast second click
<!-- AC:END -->
