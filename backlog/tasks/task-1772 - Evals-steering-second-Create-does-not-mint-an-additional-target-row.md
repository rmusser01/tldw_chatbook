---
id: TASK-1772
title: 'Evals steering: second Create does not mint an additional target row'
status: Done
assignee: []
created_date: '2026-08-01 21:10'
updated_date: '2026-08-02 23:10'
labels:
  - evals
  - test-failure
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/UI/test_evals_steering_e2e.py::test_two_ui_authored_targets_one_steered_light_up_column_mode_delta fails on dev with 'AssertionError: second Create must mint an ADDITIONAL row'. Found while running the full Tests/UI sweep for TASK-596 Phase 1; verified pre-existing by reproducing it in a checkout containing none of that branch's code. Everything else in Tests/UI passes (2486 passed, 1 skipped), so this is the only red test in the directory.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The failing test passes, or is replaced by one that pins the intended behavior
- [ ] #2 Root cause is stated: whether the second Create genuinely fails to add a row, or the test's assumption about row identity is wrong
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Closed 2026-08-02: no longer reproducible on current dev. Evidence chain, all on 0584452f4 or later: (1) passes standalone 30/30 (15 runs x2 configurations); (2) passes with its whole file and with all 378 evals UI tests together; (3) no single predecessor file induces the failure -- 139 pairwise runs with a VALIDATED detector (positive control: forcing the target assertion to fail made the harness report a polluter, so the negative is meaningful); (4) the full deterministic prefix (139 predecessor files + target, the exact context of the original sighting) passes: 2781 passed, 0 failed. Plausible fix vector: evals steering commits merged between the sighting (c11c5b199 sweep, 2026-08-01) and now, notably 022e1f017 'fix(evals): whole-branch review fix wave -- steered targets...' (task-1691 phase 2). Root cause at the original commit was never isolated; if this recurs, the polluter-finder harness is at the job dir's tmp/find-polluter.sh pattern -- pairwise predecessor+target with a harness self-check that refuses runs producing no pytest verdict (the first sweep silently no-op'd on macOS's missing 'timeout' command and reported 143 false 'clean' verdicts). pytest-randomly is NOT installed, so ordering is deterministic and results are reproducible.
<!-- SECTION:NOTES:END -->
