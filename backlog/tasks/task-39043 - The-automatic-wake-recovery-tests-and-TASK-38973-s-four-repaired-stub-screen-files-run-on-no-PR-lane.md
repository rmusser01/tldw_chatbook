---
id: TASK-39043
title: "The automatic wake recovery tests and TASK-38973's four repaired stub-screen files run on no PR lane"
status: To Do
created_date: 2026-11-06 10:09
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
- [ ] #1 `test_automatic_wake_recovery.py` and the three `Tests/Backup_Recovery` files run on a PR lane, appended at the end of "Run Dreams and Guardian boundary contracts" (the storage-pausing entries pinned first stay first), and every added file passes under `--storage-lease-guard`
- [ ] #2 `Tests/UI/test_console_full_config_preparation.py` runs whole as a UI census tail entry, and `scripts/check_ui_pr_gate_census.py` passes
- [ ] #3 The `Tests/CI` lane pins (`DREAMS_BOUNDARY_TARGETS`, the step argv pin in `test_response_rules_critical_gate.py`) are updated
- [ ] #4 Recent Dreams boundary job and step times are quoted, and the step's exact command with the additions is run three times with the times recorded
- [ ] #5 The owner decides whether the two ~86 s files (`test_console_config_sync_lifetime.py`, `test_home_citation_retirement.py`) go in the Dreams boundary step or another lane; the decision is recorded
<!-- AC:END -->
