---
id: TASK-31953
title: >-
  Library reader - a typed custom items width is still widened by the
  library-closed comfort branch
status: Done
assignee:
  - '@claude'
created_date: '2026-09-07 08:26'
updated_date: '2026-09-07 17:41'
labels:
  - library
  - media-ux
  - test-debt
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
H final review: test_a_typed_custom_items_width_is_obeyed_rather_than_grown's docstring overclaims. The pre-existing library-closed comfort branch (adaptive_reader_state.py ~295-303) still widens a typed 32 to 52 at width 100 in custom mode. Not PR H's bug, but the pin's name and docstring now read as a guarantee the code does not make, which is how the next reader change gets blamed for it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Either the comfort branch obeys a typed custom width, or the test's name and docstring state the branch it does not cover
- [x] #2 The decision and its reason are recorded in the task or beside the pin
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Test the one-line option (obey a typed width in the library-closed comfort branch) against Media 100x30 and the siblings. 2. If it is not one line with no impact, rename the pin to the branch it covers, add a characterization pin for the widening, record the decision beside the clamp.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
DOCUMENTED, not changed: two clamps widen a typed Items width (the library-closed clamp and the priority-pane clamp in adaptive_reader_state.py; typed 32 → 52 at width 100), and test_resolution_never_mutates_saved_preferences already pins typed 40 → 56, so obeying is a user-visible width change on all four readers plus pin churn — a product decision (rider filed: are custom widths floors, exact, or advisory?). The pin is now test_a_typed_custom_items_width_is_obeyed_by_the_growth_gate with a scoped docstring; test_a_typed_custom_items_width_is_still_widened_once_the_library_closes characterizes the widening; the clamp carries the reason. Also fixed here: the batch-1 floor pin's wait is anchored on the painted region (30/30). Also fixed at the final review: Tests/Live's Media geometry contract summed 2 * PANE_GRIP_WIDTH (off by eight since PR H); it now reads layout.grip_width.
<!-- SECTION:NOTES:END -->
