---
id: TASK-27019
title: Recompose census needs an anti-slack guard like the size ratchet
status: Done
assignee:
  - '@claude'
created_date: '2026-09-02 15:14'
updated_date: '2026-09-02 19:24'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Final review of the library decomposition foundation: Tests/UI/test_library_recompose_ratchet.py pins a ceiling only; headroom drift happened twice before (107->80, 74->63). Mirror test_budget_is_not_left_slack_after_a_wave.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Census pin has a slack guard with a documented tolerance
- [x] #2 Guard is mutation-tested (headroom injected -> fails)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read task-27019, the size-ratchet model test (test_budget_is_not_left_slack_after_a_wave) and the recompose census file fully.
2. Choose a documented tolerance for the census slack guard, scaled to a small integer count (63) rather than the size ratchet's line/method budgets (~44k/1300) -- the absolute tolerance does not transfer.
3. Implement test_census_pin_is_not_left_slack in Tests/UI/test_library_recompose_ratchet.py, asserting (pin - actual_count) <= tolerance.
4. Mutation-test both directions via scratch edits to the pin constant: headroom injected (pin raised beyond tolerance) -> test fails; exact/near pin -> test passes. Capture outputs, then restore the real pin.
5. Mark task-27019 Done with Implementation Notes.
6. File the settings_screen.py size-ratchet-budget-row follow-up task via the CLI, --ac repeated per criterion.
7. Run the recompose ratchet test file plus ./scripts/preflight.sh; commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added test_census_pin_is_not_left_slack to Tests/UI/test_library_recompose_ratchet.py, mirroring test_budget_is_not_left_slack_after_a_wave. Tolerance set to 5 sites (_CENSUS_SLACK_TOLERANCE), documented at the constant: below the smallest historically-observed silent drift step (6 sites, the Reader sub-state routing change), so the guard would have fired on both past drift incidents (107->80, 74->63). Unlike the size ratchet's line/method budgets, this census has no ordinary-edit noise floor to absorb -- a recompose call site is never an incidental byproduct of unrelated work -- so the tolerance is derived from this census's own drift history rather than scaled from the size ratchet's 200/10. Mutation-tested both directions via scratch edits to the pin constant (restored after each): pin=73 (slack 10) FAILED; pin=69 (slack 6, one over tolerance) FAILED; pin=68 (slack 5, exactly at tolerance) PASSED; pin=63 (slack 0, real value) PASSED. Modified: Tests/UI/test_library_recompose_ratchet.py only.
<!-- SECTION:NOTES:END -->
