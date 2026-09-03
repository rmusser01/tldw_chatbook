---
id: TASK-30041
title: Review sets - fix the dead completion gesture and stale finish footer
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-03 13:04'
updated_date: '2026-09-03 13:07'
labels:
  - library
  - media-ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique 2026-09-03 P1: check_action for library_media_next_item/prev_item gates ]/[ on browse-row adjacency and never consults the active review set, so the documented final-] completion gesture is disabled at the last browse row (and the gate misfires whenever set order diverges from browse order). The walk's clamp branch also skips the viewer sync, leaving the footer stale at the completion moment. Reproduced live; docs promise the gesture.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With an active review set, ] and [ are enabled wherever the set can move or mark, regardless of browse-row adjacency
- [ ] #2 The final ] on the last live item marks it done and the footer updates to the all-reviewed state in the same interaction
- [ ] #3 Completing a set surfaces an explicit completion notice
- [ ] #4 Walking a selection-origin set whose order differs from browse order works end to end
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. check_action ]/[ branch: enabled whenever _review_set_active() (walk clamps safely), else browse adjacency - TDD via fake\n2. Walk clamp branch: viewer sync so the footer updates at the finish - TDD\n3. Completion notice: walk that marks and flips the set to complete notifies 'All N reviewed.' - TDD\n4. Live verify: 6-item set, walk to final ], footer flips + notice
<!-- SECTION:PLAN:END -->
