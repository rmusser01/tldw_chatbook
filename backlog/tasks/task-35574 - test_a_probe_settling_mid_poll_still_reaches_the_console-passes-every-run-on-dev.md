---
id: TASK-35574
title: test_a_probe_settling_mid_poll_still_reaches_the_console passes every run on dev
status: To Do
created_date: 2026-10-19 05:30
dependencies: []
labels:
- follow-up
- tests
- console
priority: low
references:
- Tests/UI/test_console_switcher_local_probe.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in Phase 9's post-rebase interaction check (2026-10-19). `Tests/UI/test_console_switcher_local_probe.py::test_a_probe_settling_mid_poll_still_reaches_the_console` failed serially 2 of 3 times on plain dev 69771c1811, and 3 of 3 on Phase 9 rebased onto it, always with the same assertion: the Console read 'Ready · not tested' where the test expects 'Not ready · refused :9099'. It runs in no PR lane, so no PR has seen it fail. Whether the product or the test is at fault (a probe that settles mid-poll never reaching the Console, or a test that reads before the next poll) is not known yet.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The cause is found and recorded in this task: a product defect (a settled probe that never reaches the Console's readiness) or a test-timing defect
- [ ] #2 The test passes 10 of 10 serial runs on dev, and still fails when the mid-poll settle is not propagated
<!-- AC:END -->
