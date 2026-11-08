---
id: TASK-39043
title: "The automatic wake recovery tests and TASK-38973's four repaired stub-screen files run on no PR lane"
status: Done
assignee: ['@claude']
created_date: 2026-11-06 10:09
updated_date: 2026-11-08 00:45
labels:
- follow-up
- ci
- tests
- console
- backup-recovery
priority: medium
references:
- Tests/Chat/test_automatic_wake_recovery.py
- Tests/Backup_Recovery/test_console_config_sync_lifetime.py
- Tests/Backup_Recovery/test_console_sync_maintenance.py
- Tests/Backup_Recovery/test_home_citation_retirement.py
- Tests/UI/test_console_full_config_preparation.py
- .github/workflows/derived-artifacts.yml
- scripts/ui_pr_gate_census.txt
- Tests/CI/test_ci_queue_pressure_contract.py
- Tests/CI/test_response_rules_critical_gate.py
dependencies:
- TASK-38971
- TASK-38973
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 10 of the 2026-10-23 burn-down; checked on dev b2c618185f, 2026-11-06, and on send PR #3433 (head e3106a313c) and Console tests PR #3434 (head 16c99d2d9f), which land TASK-38971 and TASK-38973; the runs used those PRs stacked with storage #3435. Line numbers are dev's.

**What has no lane.**
- `grep -n <file> .github/workflows/*.yml scripts/ui_pr_gate_census.txt` finds no PR lane for any of these five files. `test_home_citation_retirement.py` appears in `perf-guard.yml:52,96` only as a `paths:` trigger.
- `Tests/Chat/test_automatic_wake_recovery.py` holds TASK-38971's new test, recovery with no agent-runs DB. Dev has 6 tests and the merge adds the 7th.
- TASK-38973 repaired four stub-screen files, which were 26 failed / 38 passed on dev and 64 passed on #3434:
  - `Tests/Backup_Recovery/test_console_config_sync_lifetime.py`;
  - `Tests/Backup_Recovery/test_console_sync_maintenance.py`;
  - `Tests/Backup_Recovery/test_home_citation_retirement.py`;
  - `Tests/UI/test_console_full_config_preparation.py`.

**Local timings on #3434** (load 16-31):

| File | `-n 4` x3 | Serial, the dreams step's flags (`--storage-lease-guard --timeout=180 --tb=short`) |
|---|---|---|
| `test_automatic_wake_recovery.py` | 7 passed, 7.3 / 7.6 / 7.5 s | 7 passed, 13.3 s |
| The four TASK-38973 files together | 64 passed, 78.8 / 79.4 / 58.2 s | see below |
| `test_console_config_sync_lifetime.py` | | 27 passed, 87.0 s |
| `test_console_sync_maintenance.py` | | 6 passed, 16.8 s |
| `test_home_citation_retirement.py` | | 24 passed, 85.9 s |
| `test_console_full_config_preparation.py` (UI-lane flags) | | 7 passed, 16.9 s |

- The four non-UI files add about 203 s of local serial time.

**Lane headroom.** In the last 12 successful PR runs (2026-11-05 23:xx to 2026-11-06 03:59, runs 42135041616 to 42159751249), the Dreams boundary job took 5.9-9.0 min of its 20 min cap. Its "Run Dreams and Guardian boundary contracts" step took 1.9-3.8 min. That step runs serially in CI. Never add to the fast PR contract (TASK-38716).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `test_automatic_wake_recovery.py` and the three `Tests/Backup_Recovery` files run on a PR lane, appended at the end of "Run Dreams and Guardian boundary contracts" (the storage-pausing entries pinned first stay first), and every added file passes under `--storage-lease-guard`
- [x] #2 `Tests/UI/test_console_full_config_preparation.py` runs whole as a UI census tail entry, and `scripts/check_ui_pr_gate_census.py` passes
- [x] #3 The `Tests/CI` lane pins (`DREAMS_BOUNDARY_TARGETS`, the step argv pin in `test_response_rules_critical_gate.py`) are updated
- [x] #4 Recent Dreams boundary job and step times are quoted, and the step's exact command with the additions is run three times with the times recorded
- [x] #5 The owner decides whether the two ~86 s files (`test_console_config_sync_lifetime.py`, `test_home_citation_retirement.py`) go in the Dreams boundary step or another lane; the decision is recorded
<!-- AC:END -->

## Implementation Plan

1. Read recent Dreams boundary job and step times (read-only `gh`).
2. Append `test_automatic_wake_recovery.py` and the small `test_console_sync_maintenance.py` to "Run Dreams and Guardian boundary contracts", before `--storage-lease-guard`; update `DREAMS_BOUNDARY_TARGETS` and the step argv pin.
3. Put the two ~86 s files in the new `recovered-suites` job (owner decision 2026-11-06) and `test_console_full_config_preparation.py` whole on the UI census tail.
4. Time the Dreams step 3 times with `jobstep.sh`; check the census.

## Implementation Notes

**Owner decision (2026-11-06):** "New CI job: add one derived-artifacts job, 'PR lane (recovered suites)', holding all of these files serially under --storage-lease-guard; one more runner per PR; ADR-103's job and runner counts updated. The Dreams job keeps its margin; only TASK-39043's two small files go there." So AC#5's two ~86 s files (`test_console_config_sync_lifetime.py`, `test_home_citation_retirement.py`) run first in the new `recovered-suites` job, which uses the same guard; AC#1's Dreams placement holds for the other two files.

**AC#1, AC#3.** `Tests/Backup_Recovery/test_console_sync_maintenance.py` (sandboxed) and `Tests/Chat/test_automatic_wake_recovery.py` (bootstrap_profile) are the last targets of "Run Dreams and Guardian boundary contracts", after `Tests/DB/test_sync_log_retention.py` and before `--storage-lease-guard`; the storage-pausing file stays first. The sandboxed file follows three bootstrap_profile targets, which TASK-32873's note warns about, so it was checked there: every run of the step in that order was green. `DREAMS_BOUNDARY_TARGETS` and the argv pin in `Tests/CI/test_response_rules_critical_gate.py` name both. The two ~86 s files passed in every recovered-suites run under the guard (27 and 24 tests).

**AC#2.** `Tests/UI/test_console_full_config_preparation.py` runs whole as a UI census tail entry: 7 passed, about 25 s serially with the UI step's command (25.0 s at load 50-155, 26.2 s at load about 160). It is bootstrap_profile (module pytestmark), as 512 census entries already were; it first ran in the recovered-suites step because I wrongly read the census as unable to hold such files, and review r12l-rev1 corrected that. `scripts/check_ui_pr_gate_census.py`: OK, 1870 entries. UI shard 34, which holds it in the new deal, ran in lane order: 109 passed in 895 s at load 36-58.

**AC#4.**

**CI before (read-only `gh`).** Dreams boundary job over 15 PR runs, 2026-11-06 14:36Z to 2026-11-07 00:55Z (42235403758 to 42321107301): job 9.80-14.05 min, step 6.10-8.93 min. None of them has round 11's three Dreams additions; no PR run of round-11 dev had finished when these were read (the queue held 04:33Z-05:31Z runs, all queued or cancelled). The recovered-suites job is new, so it has no before time.

**The Dreams step**, `jobstep.sh fu-r12-lanes "Run Dreams and Guardian boundary contracts"` (33 targets), 3 runs on 2026-11-07: 548.1 s (load 52 falling to 12), 422.4 s (load 12-8), 372.6 s (load 8-5); 1022 passed each, no lease finding. A diagnostic run of the same command with full output, at load 49-67, took 1214 s, also all green. The two TASK-39043 files take about 30 s of it serially (13 s and 17 s on #3434).

Projected for CI: the round-11 notes put the job at about 13.2-16.8 min once round 11's files land; these two add about 0.5 min.

**The recovered-suites step**, `jobstep.sh fu-r12-lanes "Run the recovered suites"` (14 targets, serial, `--storage-lease-guard --timeout=180`), 3 runs on 2026-11-07: 534.8 s (load 40-63), 703.2 s (load 40-60), 641.7 s (load 50-60). Each run: 723 passed, 1 skipped, 2 failed. The 2 are `test_executor_failure_uses_stable_job_terminal[artifact_incompatible-failed]` and `[engine_crashed-failed]` in the ingest runner, TASK-39140's stale `error_detail` pin, red on dev too and fixed on round-12 branch fu/r12-ctlr (489e2338f5); the step is green once both land. No lease finding in any run. The slowest run plus about 1.5 min of setup is about 13.2 min, 6.8 min under the job's 20-min cap; CI ran comparable lanes at 0.5-1.0 times this machine's loaded time.

**Files.** `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`, `Tests/CI/test_response_rules_critical_gate.py`, `Tests/CI/test_derived_artifacts_workflow.py`.

**Review r12l-rev1 (2026-11-08).** The two bootstrap_profile Tests/UI files left the recovered-suites step for the UI census (the census does hold bootstrap_profile files: 512 entries already, and TASK-36478 restores the shared bootstrap config after each test; my earlier "may not hold" was wrong). TASK-39124's built-in pixel buddy file joined the step last. The final step, 13 targets, `jobstep.sh fu-r12-lanes "Run the recovered suites"`, 3 runs: 363.3 s (load 38 falling to 21), 742.9 s (load 21-50), 467.1 s (load 50-43); each 715 passed, 1 skipped, 2 failed (TASK-39140's pin, fixed on fu/r12-ctlr), no lease finding. The slowest run plus about 1.5 min of setup is about 13.9 min, 6.1 under the cap.
