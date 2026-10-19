---
id: TASK-35575
title: The admission-sensitive PR job keeps room for a hung test's report under its cap
status: In Progress
created_date: 2026-10-19 05:40
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

Serial in the job's install set, the 23 took 15.7 minutes and the whole step 29.1 minutes at load 23-57; at load 10-27 they took 9.6 and 18.7 minutes. Scaling the whole step by the 23's own local-to-CI ratio (the pytest step's CI minutes, TASK-34916: 15.4 median, 18.7 max), plus setup (about 0.8 minutes at the median, at least 1.4 in the slowest run), gives about median 29-31 / max 33-38 minutes on CI: 33-35 if a slow run's extra minutes are the job's own fixed 3.8-minute spread, 36-38 if it is slow throughout. So a slow run can reach the cap with no hung test, and one hung test at `--timeout=300` (about 5.3 minutes with the private-profile wrapper's report) passes it and loses that report (the #2992 failure mode). The workflow's own rule for a lane near its cap is to add a job, not minutes. Either choice changes something the owner owns: a second job is one more runner per PR (ADR-103's counts), and a timeout change revises the owner's 35.

The PR's own CI runs give the real number; record them here.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The owner's decision is recorded in this task: a second admission-sensitive job, a timeout change, or moving targets out under the rule TASK-35570 decides
- [ ] #2 After the change, the slowest of at least 5 measured PR runs of each admission-sensitive job ends at least 8 minutes under its cap, which leaves room for one hung test's report
- [x] #3 Tests/CI's lane contracts and ADR-103's runner note match the jobs as changed
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**AC#1, the decision (owner, asked directly, 2026-10-19):** a second admission-sensitive job. With Phase 9's 52 targets the one job took 29.8 and 27.4 min on PR #3169's two CI runs (the pytest step 1737 s and 1590 s) against its 35-min cap, so one hung test's ~5.3-minute report would have passed the cap.

**The change (Phase 9 round-2 review, W1).** `pr-fast-lane-admission` ("PR Fast Lane (admission-sensitive)") keeps `dev`'s 41 targets in `dev`'s order, unchanged, and its job id and name stay. A new job, `pr-fast-lane-admission-2` ("PR Fast Lane (admission-sensitive 2)"), has the same `runs-on`, `if:`, `timeout-minutes: 35`, checkout, setup-python and install steps, and one pytest step, "Run admission-sensitive suites 2", with Phase 9's 52 targets in their order there (the three many-case rule tests first) and `--timeout=300`. The required aggregate `needs` it, and a "Require successful PR fast lane (admission-sensitive 2)" step fails it as the other lanes do. Both jobs' header comments give the figures below.

**Measured.** `dev`'s 41 alone over 20 PR runs on CI (2026-10-18 21:36 to 10-19 11:11 UTC; each head's workflow checked to hold the 41): job median 17.5 / max 20.0 min, the pytest step median 996 s / max 1147 s, so a hung test's report still ends about 10 min under the cap. Phase 9's 52 alone, serial in the job's install set (`--timeout=300`, load 7-9): 959 passed in 266.9 s, where all 93 took 1744.6 s locally and 1737 s and 1590 s on CI. So expect about 5 min of pytest and 6-7 min with setup.

**AC#3.** Tests/CI: `ADMISSION_2_TARGETS` with its `EXACT_STEPS` entry (so the selection-flag, shorter-limit and missing-target mutations cover it), `test_second_admission_target_set_is_exact` (the three rule tests lead; no target shared with the first job, a node beside its whole file included), the aggregate's `needs`, verdict and continue-on-error contracts, `test_required_aggregation_contract_rejects_a_dropped_admission_lane`, the one-serial-job contract, the Console P0 gate's own-targets check, `test_derived_artifacts_workflow.py`'s job list, verdicts and disjoint-targets step list, and the P0 gate's `_lanes()` and `_BOOTSTRAP_OK`. ADR-103 has a dated note, and its runner counts are now sixteen, seventeen with perf-guard (`test_ui_lane_shard_count_statements.py` failed on the old counts until then). Negative controls on the real workflow: dropping one target from the new job fails `test_second_admission_target_set_is_exact`; dropping one from the first job fails `test_admission_target_set_is_exact` (and, for a Notes sync file, the real-stack test); dropping the new job from the aggregate's `needs` fails six tests, `test_required_aggregation_contract_rejects_a_dropped_admission_lane[pr-fast-lane-admission-2]` among them. Tests/CI: 1015 passed, 1 skipped.

**AC#2 stays open** until five PR runs of each job are measured.

Files: .github/workflows/derived-artifacts.yml, Tests/CI/test_ci_queue_pressure_contract.py, Tests/CI/test_derived_artifacts_workflow.py, Tests/CI/test_console_p0_regression_gate.py, backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md.
<!-- SECTION:NOTES:END -->
