---
id: TASK-21592
title: >-
  Tests App pollutes Tests Library in the same process   13 to 15 errors
status: To Do
assignee: []
created_date: '2026-08-23'
labels:
  - testing
  - flaky
  - test-isolation
priority: medium
---
## Description

Running `Tests/App` and `Tests/Library` in the same process produces 13-15 errors whose node ids
vary run to run, and zero errors when either directory runs alone. This is cross-file state
leakage, and it is a real source of CI flakiness that will misattribute failures to whichever
branch happens to be running.

## Acceptance Criteria

- [ ] The leaking state is identified by name (module-level singleton, patched global, app instance, or fixture scope) rather than worked around with ordering or `-p no:randomly`
- [ ] `Tests/App` and `Tests/Library` run clean together in one process, repeatedly and under random ordering
- [ ] A guard prevents the same class of leak from returning — e.g. an autouse fixture asserting the relevant global is unset at teardown
- [ ] The error count is confirmed to be zero on several consecutive runs, since the symptom is nondeterministic

## Evidence

Observed by the TASK-21111 implementer while A/B-baselining: 13 errors on its branch and 15 of
the same class on pristine dev `f49956038`, with varying node ids; running its new file alone with
those Library files gave 306 passed and 0 errors.

Pre-existing and unrelated to that task's change.
