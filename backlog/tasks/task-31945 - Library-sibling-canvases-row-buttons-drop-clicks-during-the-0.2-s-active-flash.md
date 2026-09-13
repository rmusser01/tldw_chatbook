---
id: TASK-31945
title: >-
  Library sibling canvases - row buttons drop clicks during the 0.2 s active
  flash
status: Done
assignee: []
created_date: '2026-09-07 08:25'
updated_date: '2026-09-07 20:03'
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
- [x] #1 A second click on a sibling-canvas row button within 0.2 s of the first is delivered
- [x] #2 All four library canvases share one row-button press behaviour
- [x] #3 At least one sibling canvas has a pin for the fast second click
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Pin on a sibling canvas: two presses on a row button inside 0.2 s are both delivered. 2. One shared row-button construction helper carrying `active_effect_duration = 0` for the conversations, notes, prompts and skills row buttons.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Textual's `Button._on_click` drops a click inside the previous press's 0.2 s `-active` flash; PR F set `active_effect_duration = 0` on the Media rows. One shared helper now builds the conversations, notes, prompts and skills row buttons with it; trash, collections and rail rows are not routed (different press semantics, no fast-second-click flow). Pin on a Conversations row (verified red before the fix); live: marker-then-title clicks with no delay toggled 1→0→1. Side effect: four pre-existing reds in the prompts test file went green (they were hitting the dropped second click).
<!-- SECTION:NOTES:END -->
