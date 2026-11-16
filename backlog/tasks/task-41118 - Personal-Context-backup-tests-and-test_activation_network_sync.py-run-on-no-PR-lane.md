---
id: TASK-41118
title: "Personal Context backup tests and test_activation_network_sync.py run on no PR lane"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-15 20:07'
created_date: 2026-11-15 02:20
labels:
- follow-up
- backup
- personal-context
- tests
- lanes
priority: medium
references:
- Tests/Backup_Recovery/test_activation_network_sync.py
- Tests/Backup_Recovery/test_portable_relocation.py
- Tests/Backup_Recovery/test_restore_data_groups.py
- Tests/Personal_Context/test_service.py
- .github/workflows/derived-artifacts.yml
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** Round 14 (TASK-40133..40140, 40711) put the Personal Context backup, restore, re-encryption and relink behaviour under tests that no PR job runs, so a regression lands on dev unseen.

**Evidence.** None of these files is named in `.github/workflows/*.yml` or the UI census at the stack top: `Tests/Backup_Recovery/test_activation_network_sync.py`, `test_portable_relocation.py`, `test_restore_data_groups.py`, `test_credentials.py`, `test_data_groups.py`, `test_inert_extraction.py`, `Tests/Personal_Context/test_repository.py`, `test_runtime_policy.py`, `test_service.py`, `Tests/App/test_personal_context_wiring.py`. `test_detached_layout.py` runs whole only in the push-to-dev owner job; TASK-40138's notes list its new nodes as unlaned on PRs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Personal Context backup and restore tests added in round 14 run on a PR lane or the push-to-dev owner job (never the fast PR contract), chosen per file and recorded with each file's measured time
- [x] #2 test_activation_network_sync.py runs on such a lane
- [x] #3 The lane targets are pinned in Tests/CI
<!-- AC:END -->

## Implementation Plan

1. Measure each listed Personal Context / backup file at the lane's flags.
2. Lane each file on the push-to-dev owner job or an existing PR step (never the fast PR contract), per file, with its time recorded.
3. Pin the targets in Tests/CI and update ADR-103 where counts change.

## Implementation Notes

Commit 7d9c3988b1. All on the push-to-dev owner job (`backup-owner-suites-dev`, -n 4, no `--storage-lease-guard`), pinned in `BACKUP_OWNER_SUITE_TARGETS`.

Per file, measured serially with recovered-suites' flags (`-p no:xdist --timeout=180 --storage-lease-guard`) unless stated, on a shared 18-core host:
| file | result | time | load |
|---|---|---|---|
| Backup_Recovery/test_activation_network_sync.py | 50 passed | 346 s | 79 |
| Backup_Recovery/test_portable_relocation.py | 14 passed | 326 s | 73 |
| Backup_Recovery/test_restore_data_groups.py | -n 4: 36 passed, 4 pytest-timeout (>300 s) | 12.5 min at -n 4 | ~110 |
| Backup_Recovery/test_credentials.py | 91 passed | 203 s | 123 |
| Backup_Recovery/test_data_groups.py | 29 passed | 29 s | 107 |
| Backup_Recovery/test_inert_extraction.py | 25 passed (-n 4) | 35 s at -n 4 | ~85 |
| Personal_Context/test_repository.py | 78 passed, 23 lease-guard errors | 249 s | 118 |
| Personal_Context/test_runtime_policy.py | 14 passed, 2 lease-guard errors | 72 s | 115 |
| Personal_Context/test_service.py | 49 passed, 7 lease-guard errors | 281 s | 123 |
| App/test_personal_context_wiring.py | 9 passed, 1 lease-guard error | 27 s | 110 |

Choice: the owner job, not recovered-suites. The four Personal Context/App files error under `--storage-lease-guard` (leaked leases; without the guard, -n 4: 150 passed in 134 s at load 85), and recovered-suites already runs 12-15 of its 20 CI minutes; the Backup_Recovery files are heavy child-process suites like the job's others. The job's caps rise by 20 min (step 70, job 75); ADR-103's dated paragraph records it. No PR runner or job is added.

Concern: `test_restore_data_groups.py`'s `test_replace_restore_makes_personal_context_readable_and_rolls_back` cases took 83-300+ s each at load ~110; four hit the 300 s limit here. They pass in under 300 s when the host is less loaded, but the first CI runs of the job should be read for them.
`test_detached_layout.py` already runs whole in the same job.
