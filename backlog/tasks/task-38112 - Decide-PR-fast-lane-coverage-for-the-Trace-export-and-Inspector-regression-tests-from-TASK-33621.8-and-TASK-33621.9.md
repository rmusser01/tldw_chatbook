---
id: TASK-38112
title: Decide PR fast-lane coverage for the Trace export and Inspector regression
  tests from TASK-33621.8 and TASK-33621.9
status: To Do
assignee: []
created_date: 2026-10-30 05:56
labels:
- tests
- ci
- owner-decision
- follow-up
dependencies: []
priority: low
updated_date: 2026-11-01 17:59
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up from TASK33621.8/.9: decide remaining PR fast-lane coverage for the Chat staged-evidence export, native-reader usage and stored-orphan export regressions, and verify lane timing. PR3321 backlog integration appends eight verified Inspector/export UI selectors after the complete existing census prefix; the earlier blanket statement that none of the new UI regressions are selected is superseded. The Chat regression selection and the original30-second whole-file performance criterion remain unqualified. No additional hosted-CI policy change is authorized by this task.
<!-- SECTION:DESCRIPTION:END -->
## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The owner has decided which of the TASK-33621.8 and TASK-33621.9 regression tests run on PRs, and the decision is recorded in this task
- [ ] #2 Any test added to the PR gate runs under 30 seconds per file and passes under the plain command
- [ ] #3 scripts/check_ui_pr_gate_census.py passes after the change
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-11-01 integration update: eight UI selectors appended, final census1311 retains all1303 prior entries and passes the canonical checker. Focused local final-base Trace/Inspector checks pass168 cases. This is functional coverage evidence, not a30-second whole-file timing qualification; remaining Chat lane selection/timing decisions keep this task To Do.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
