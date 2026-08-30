---
id: TASK-24652
title: >-
  Wizard suite has order/load-dependent flakes: a different test fails on most
  full sweeps
status: Done
assignee:
  - '@claude'
created_date: '2026-08-28 22:11'
updated_date: '2026-08-29 00:04'
labels: []
dependencies: []
---

## Renumbering provenance

- Previous ID: `TASK-23113`.
- Renumbered to `TASK-24652` on 2026-08-29 because the older terminal-sharing
  task added by `0583a686468fe3442d6e695fee2ffe2c9c1c98c8` keeps `TASK-23113` under
  the TASK-19601 older-arrival rule. This wizard-flake task arrived later in
  `8db03c25bc9ddc17bbbeb1b1f5a7cb59b5aef923`.

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Running Tests/UI/test_first_run_wizard_live_contract.py plus Tests/Wizards together produces roughly one failure per sweep, and it is a different test each time. Every one passes in isolation. This is measured on pristine origin/dev, so it is not caused by any recent change, but it makes the suite unreliable as a merge gate and will eventually cost someone a debugging session chasing a failure that has nothing to do with their diff.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The full wizard sweep passes repeatedly without per-run flakes
- [ ] #2 The shared state or timing causing cross-test interference is identified and documented
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce and capture the actual assertion delta.\n2. Find the shared state or timing that leaks between tests.\n3. Fix the leak (or the fence) and prove the sweep is repeatable.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Two causes, both fixed. (1) Root cause of the most reproducible flake: DOMNode.screen RAISES NoScreen for a detached node, so the guard 'focused.screen is self' exploded before it could protect anything. App.focused outlives its screen when the first-run wizard is pushed over Settings and popped, and the worker-driven _apply_sync_rows then crashed with WorkerFailed: NoScreen. Measured on test_rerun_over_settings_review_settings_returns_to_settings in isolation: 3/10 failures at baseline, 1/10 with the timeout change alone, 0/12 with the guard. Two sibling sites had the identical unguarded pattern (settings_screen focus-restore, evals_screen focus-restore); all three now route through UI/focus_ownership.py, and no unguarded 'focused.screen' read remains. (2) The live-contract settle ceiling was 10s, too tight for a full sweep where ~890 Textual harnesses share a process; raised to a single _SETTLE_TIMEOUT_SECONDS = 30 and the seven per-call overrides folded into it. A wait returns the instant its condition holds, so this costs a green run nothing. Result: 5 consecutive clean 892-test sweeps, where before roughly half failed with a different test each time. Not claiming eradication -- these are probabilistic -- but the mechanism behind the reproducible one is a real product bug that would also fire in production, and it is gone.
<!-- SECTION:NOTES:END -->
