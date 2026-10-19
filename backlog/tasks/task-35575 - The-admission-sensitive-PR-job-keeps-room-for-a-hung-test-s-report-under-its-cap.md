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
Found in Phase 9's fix-round review (2026-10-19). The PR Fast Lane's admission-sensitive job (`pr-fast-lane-admission`, a required check, `timeout-minutes: 35`, the cap the owner set in TASK-33910.2) has nearly run out of room. Its 90 measured runs to 2026-10-14 replayed at median 16.3 / max 20.1 minutes, when the step had 23 targets. Phase 8 then added 18 targets and Phase 9 added 52.

Serial in the job's install set, the 23 took 15.7 minutes and the whole step 29.1 minutes at load 23-57; at load 10-27 they took 9.6 and 18.7 minutes. Scaled by the 23's own local-to-CI ratio, the step comes to about median 25-31 / max 29-35 minutes on CI. One hung test at `--timeout=300` adds about 5.3 minutes with the private-profile wrapper's report, so a slow run with one hang can pass the cap and lose that report (the #2992 failure mode). The workflow's own rule for a lane near its cap is to add a job, not minutes. Either choice changes something the owner owns: a second job is one more runner per PR (ADR-103's counts), and a timeout change revises the owner's 35.

The PR's own CI runs give the real number; record them here.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The owner's decision is recorded in this task: a second admission-sensitive job, a timeout change, or moving targets out under the rule TASK-35570 decides
- [ ] #2 After the change, the slowest of at least 5 measured PR runs of each admission-sensitive job ends at least 8 minutes under its cap, which leaves room for one hung test's report
- [ ] #3 Tests/CI's lane contracts and ADR-103's runner note match the jobs as changed
<!-- AC:END -->
