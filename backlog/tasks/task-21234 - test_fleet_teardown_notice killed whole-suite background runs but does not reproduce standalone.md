---
id: TASK-21234
title: >-
  test_fleet_teardown_notice killed whole-suite background runs but does not
  reproduce standalone
status: To Do
assignee: []
created_date: '2026-08-23'
labels:
  - test-health
  - dev-red
  - needs-owner
dependencies: []
priority: high
---

## Description

Source: close-out of the 2026-08-22 holistic performance review burn-down; observed during
TASK-21100's fix round.

During the burn-down, `Tests/Chat/test_fleet_teardown_notice.py` hung for **more than 420 s
under a 60 s per-test timeout on pristine dev**, and was identified as the cause of
whole-suite background-run kills. It was excluded identically on both sides of every A/B so
that attribution stayed sound, which is why the burn-down's own evidence was unaffected.

Re-probed at close-out on dev `b2b1e2e0d` with a hard 180 s wall: the whole file **passed —
6 passed in 6.69 s**. The file itself has not changed since `f2e274993` (2026-08-22), i.e.
before the hang was observed, so the difference is not in the test source.

The hang is therefore conditional on something a standalone run does not reproduce — suite
ordering, a co-imported module, or concurrency. That is precisely the shape that keeps costing
whole-suite runs and that a green standalone run cannot close. It needs an owner with the
whole-suite context. A second defect is visible in the original observation and should not be
lost: a 60 s per-test timeout that lets a test run past 420 s is not doing its job.

## Acceptance Criteria

- [ ] The condition under which the file hangs is reproduced deterministically, or the hang is shown not to reproduce under a full whole-suite run on current dev and any exclusion for it is removed
- [ ] If it reproduces, the blocking wait is identified and the test no longer exceeds the per-test timeout
- [ ] A whole-suite run completes without a background kill attributable to this file
- [ ] The per-test timeout terminates this test if it hangs again, or the reason it cannot is recorded
