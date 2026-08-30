---
id: TASK-23201
title: >-
  Console: two Context rail paint tests depend on the pre-TASK-23193 section
  layout
status: To Do
assignee: []
created_date: '2026-08-30 01:06'
labels:
  - console
  - tests
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
test_conversation_status_row_label_and_value_are_separate_visual_runs and test_narrow_details_rail_paints_full_private_scratch_value read pixels out of the whole-screen compositor through a harness that does not reproduce the real app's layout. Both broke when TASK-23193 changed which rail sections ship open. The behaviour they guard is intact in the real app - the 2026-08-29 UAT captures show the Sessions scope row painting 'Conversation  None' correctly at 160x48 and 200x60 - so this is test infrastructure, not a user-visible regression. They are marked xfail until reworked onto a deterministic idiom.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both tests assert their property without reading whole-screen compositor pixels, or reproduce the real app's layout deterministically
- [ ] #2 The xfail markers are removed
<!-- AC:END -->
