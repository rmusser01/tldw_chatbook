---
id: TASK-23113
title: >-
  Wizard suite has order/load-dependent flakes: a different test fails on most
  full sweeps
status: To Do
assignee: []
created_date: '2026-08-28 22:11'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Running Tests/UI/test_first_run_wizard_live_contract.py plus Tests/Wizards together produces roughly one failure per sweep, and it is a different test each time. Every one passes in isolation. This is measured on pristine origin/dev, so it is not caused by any recent change, but it makes the suite unreliable as a merge gate and will eventually cost someone a debugging session chasing a failure that has nothing to do with their diff.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The full wizard sweep passes repeatedly without per-run flakes
- [ ] #2 The shared state or timing causing cross-test interference is identified and documented
<!-- AC:END -->

## Observed (2026-08-28)

Four distinct tests each failed exactly once across sweeps of the same two
paths, and each passed alone immediately afterwards:

- `Tests/Wizards/test_first_run_setup_wizard.py::test_mounted_model_owner_timeout_fences_late_result_and_keeps_manual_retry`
- `Tests/UI/test_first_run_wizard_live_contract.py::test_recovery_save_failure_reprompts_then_succeeds_once[False]`
- `Tests/Wizards/test_first_run_setup_wizard.py::TestComposeCrashPolicy::test_optional_step_failure_is_removed_and_reported_in_summary` (twice)
- `Tests/UI/test_first_run_wizard_live_contract.py::test_rerun_over_settings_review_settings_returns_to_settings` (twice; also on record from the original UAT)

Control: the identical sweep on a detached, pristine `origin/dev` failed the
same way (`TestComposeCrashPolicy::test_optional_step_failure...`, 890
passed). So this is inherent to the suite, not to any branch under review.

Several of the affected tests are timing-fenced (explicit timeouts, worker
settle waits), which is consistent with load sensitivity when ~890 Textual
app harnesses run in one process.
