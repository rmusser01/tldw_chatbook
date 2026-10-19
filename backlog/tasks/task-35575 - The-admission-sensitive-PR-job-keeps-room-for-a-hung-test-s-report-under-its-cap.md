---
id: TASK-35575
title: The admission-sensitive PR job keeps room for a hung test's report under its cap
status: To Do
created_date: 2026-10-19 06:00
dependencies: []
labels:
- model-config-redesign
- follow-up
- ci
priority: high
references:
- .github/workflows/derived-artifacts.yml
- Tests/CI/test_ci_queue_pressure_contract.py
- Tests/CI/test_console_p0_regression_gate.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in Phase 9's fix-round review (2026-10-19). The PR Fast Lane's admission-sensitive job (`pr-fast-lane-admission`, a required check, `timeout-minutes: 35`, the cap the owner set in TASK-33910.2) has nearly run out of room. Its 90 measured runs to 2026-10-14 replayed at median 16.3 / max 20.1 minutes, when the step had 23 targets. Dev then added 18 targets (17 from Phase 8 on 2026-10-17/18 UTC, 1 from a canvas fix on 10-18), and Phase 9 added 52.

Serial in the job's install set, the 23 took 15.7 minutes and the whole step 29.1 minutes at load 23-57; at load 10-27 they took 9.6 and 18.7 minutes. Scaling the whole step by the 23's own local-to-CI ratio (the pytest step's CI minutes, TASK-34916: 15.4 median, 18.7 max), plus about 0.8 minutes of setup, gives about median 29-31 / max 33-37 minutes on CI: 33-34 if a slow run's extra minutes are a fixed delay, 35-37 if it is slow throughout. So a slow run can reach the cap with no hung test, and one hung test at `--timeout=300` (about 5.3 minutes with the private-profile wrapper's report) passes it and loses that report (the #2992 failure mode). The workflow's own rule for a lane near its cap is to add a job, not minutes. Either choice changes something the owner owns: a second job is one more runner per PR (ADR-103's counts), and a timeout change revises the owner's 35.

The PR's own CI runs give the real number; record them here.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The owner's decision is recorded in this task: a second admission-sensitive job, a timeout change, or moving targets out under the rule TASK-35570 decides
- [ ] #2 After the change, the slowest of at least 5 measured PR runs of each admission-sensitive job ends at least 8 minutes under its cap, which leaves room for one hung test's report
- [ ] #3 Tests/CI's lane contracts and ADR-103's runner note match the jobs as changed
<!-- AC:END -->
