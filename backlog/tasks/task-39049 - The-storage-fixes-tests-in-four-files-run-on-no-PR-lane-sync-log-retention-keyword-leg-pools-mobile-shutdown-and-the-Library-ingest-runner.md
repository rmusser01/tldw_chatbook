---
id: TASK-39049
title: "The storage fixes' tests in four files run on no PR lane: sync-log retention, keyword-leg pools, mobile shutdown and the Library ingest runner"
status: To Do
created_date: 2026-11-06 10:09
labels:
- follow-up
- ci
- tests
- database
priority: medium
references:
- .github/workflows/derived-artifacts.yml
- Tests/CI/test_ci_queue_pressure_contract.py
- Tests/CI/test_response_rules_critical_gate.py
- Tests/DB/test_sync_log_retention.py
- Tests/RAG_Search/test_keyword_leg_db_resolution.py
- Tests/Web_Server/test_mobile_service.py
- Tests/Library/test_library_ingest_runner.py
dependencies:
- TASK-38954
- TASK-38955
- TASK-38956
- TASK-38972
- TASK-38992
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 10 of the 2026-10-23 burn-down; checked on dev b2c618185f, 2026-11-06, and on the state after the storage PR (TASK-38954, 38955, 38956, 38972, 38986) merges. Line numbers are from that post-merge state.

**What has no lane.**
- The storage PR adds or changes tests in eight places. These are laned:
  - the three Guardian files TASK-38986 fixed (dreams-boundary step, `derived-artifacts.yml:1106-1110`);
  - `Tests/CI` and the module-size ratchet (fast PR contract);
  - `Tests/Performance/test_boot_worker_census.py` (`perf-guard.yml:180`).
- These four run on no PR lane, either after the merge or on dev; grep of `derived-artifacts.yml`, `perf-guard.yml` and `scripts/ui_pr_gate_census.txt`:
  - `Tests/DB/test_sync_log_retention.py`: TASK-38955's quit-time retirement and its retention-pass worker handles.
  - `Tests/RAG_Search/test_keyword_leg_db_resolution.py`: TASK-38956's FTS5 pool retirement.
  - `Tests/Web_Server/test_mobile_service.py`: TASK-38972's mobile shutdown.
  - `Tests/Library/test_library_ingest_runner.py`: TASK-38954's Research startup ordering. Three of its nodes run only in the label-triggered `task-603-platform-evidence.yml`.

**Local timings after the merge** (pytest time; 3 runs at `-n 4`, then 1 serial run with the dreams step's flags `-p no:xdist --storage-lease-guard --timeout=180`; load 17-30):

| File | Result locally | `-n 4` x3 | Serial with the guard |
|---|---|---|---|
| `test_sync_log_retention.py` | 23 passed | 9.6 / 7.8 / 8.8 s | 10.6 s, guard-clean |
| `test_keyword_leg_db_resolution.py` | 11 failed, all `RecoveryRequired` (TASK-38111) | 17.1 / 17.1 / 15.2 s | 4.0 s |
| `test_mobile_service.py` | 32 passed, 1 failed | 24.5 / 22.2 / 21.0 s | 76.2 s, guard-clean |
| `test_library_ingest_runner.py` | 70 passed, 86 failed, 1 skipped | 31.2 / 30.0 / 30.4 s | 61.2 s, **130 lease-guard errors** (127 `library_ingest.db` on MainThread, the rest `workspace.sqlite`) |

- Of the 98 local failures, 96 are `RecoveryRequired: raw_source_selection_changed` (TASK-38111, the local trap). The others are:
  - `test_real_app_wires_research_association_and_restores_before_startup_resume`, which follows the "staging unavailable" path TASK-38954's notes describe;
  - `test_serve_wiring_attaches_relay_only_when_broker_exists` (`AttributeError … revoke_browser_session`), which TASK-38992 owns.
- The local times for the two RecoveryRequired files are too low, because their tests fail at setup here.

**Lane headroom.** The dreams-boundary step ("Run Dreams and Guardian boundary contracts", 20 min cap, serial in CI) took 3m00s-4m02s in five recent PR runs (42178132011, 42160401969, 42161903117, 42162010199, 42159751249); its job took 6m30s-8m54s. Never add to the fast PR contract (TASK-38716).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `Tests/DB/test_sync_log_retention.py` runs whole on a PR lane, appended at the end of the dreams-boundary step (never ahead of the storage-pausing tests pinned first); the lane pins in `Tests/CI` (`DREAMS_BOUNDARY_TARGETS`, the step argv pin in `test_response_rules_critical_gate.py`) are updated, and three timings of the step's exact command are recorded
- [ ] #2 `Tests/Web_Server/test_mobile_service.py` runs on a PR lane once TASK-38992 makes it green, with three timings of that lane's command recorded
- [ ] #3 `Tests/RAG_Search/test_keyword_leg_db_resolution.py` runs on a PR lane; its evidence is a green CI run of that lane (locally every node hits TASK-38111), linked in the notes
- [ ] #4 `Tests/Library/test_library_ingest_runner.py` either leaves no live lease (0 lease-guard errors under `--storage-lease-guard`, with the before count of 130 recorded) and joins the guarded step, or the owner places it on a lane without the guard; the decision is recorded
- [ ] #5 Recent job times of every lane touched are quoted before and after; the fast PR contract is not used
<!-- AC:END -->
