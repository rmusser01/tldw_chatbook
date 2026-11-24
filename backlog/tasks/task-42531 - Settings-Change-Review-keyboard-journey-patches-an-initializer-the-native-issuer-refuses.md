---
id: TASK-42531
title: Settings Change Review keyboard journey patches an initializer the native issuer refuses
status: Done
assignee:
  - '@claude'
created_date: '2026-11-24 08:00'
updated_date: '2026-11-24 08:00'
labels:
  - tests
  - workspaces
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The light CI tier (TASK-42521) ran `Tests/UI/test_settings_change_review_journeys.py` whole on #3688, and all four `test_change_review_keyboard_conflict_preparation_and_retry` cases failed. They fail the same way on dev, on macOS and on Linux, and have since the #3480 merge (e10ce63fe8, 2026-11-12). No PR lane runs the file, so nothing noticed. Any PR that touches it now stops on these four.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All four keyboard-journey cases pass on dev's code with their cause recorded
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm the commit that turned the cases red and why.
2. Move the test's hold onto a seam the current product still calls.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Cause: a stale test seam.** The cases passed at #3480's first parent (7775edb567) and fail at its merge (e10ce63fe8).

- The test held preparation by patching `service._initialize_root` after the app was wired.
- Since #3480, the native issuer captures `service._initialize_root` at wiring (`Workspaces/change_tracking.py`) and treats a different one as a changed source (`check()` raises "native initialization source changed").
- So preparation failed before the test's initializer ran, and the test's `entered.wait(2)` timed out.

The product behaviour is correct: a replaced initializer is a changed source.

**Fix.** The test now holds `change_turn_tracker.initialize_shadow_root`, the step the default initializer calls, and takes the `service=` keyword it is passed. The journey itself is unchanged.

**Evidence.** On macOS the file goes from 4 failed / 4 passed to 8 passed.
<!-- SECTION:NOTES:END -->
