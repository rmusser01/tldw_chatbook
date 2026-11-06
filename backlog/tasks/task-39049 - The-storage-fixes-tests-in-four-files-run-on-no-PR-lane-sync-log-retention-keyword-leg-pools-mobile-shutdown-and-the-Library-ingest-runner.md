---
id: TASK-39049
title: "The storage fixes' tests in four files run on no PR lane: sync-log retention, keyword-leg pools, mobile shutdown and the Library ingest runner"
status: In Progress
assignee: ['@claude']
created_date: 2026-11-06 10:09
updated_date: 2026-11-06 17:07
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
Found in round 10 of the 2026-10-23 burn-down; checked on dev b2c618185f, 2026-11-06, and on storage PR #3435 (head 3cc391fadc), which lands TASK-38954, 38955, 38956, 38972 and 38986. Line numbers are dev's.

**What has no lane.**
- The storage PR adds or changes tests in eight places. These are laned:
  - the three Guardian files TASK-38986 fixed (dreams-boundary step, `derived-artifacts.yml:1103-1107`);
  - `Tests/CI` and the module-size ratchet (fast PR contract);
  - `Tests/Performance/test_boot_worker_census.py` (`perf-guard.yml:180`).
- These four run on no PR lane, on #3435 or on dev; grep of `derived-artifacts.yml`, `perf-guard.yml` and `scripts/ui_pr_gate_census.txt`:
  - `Tests/DB/test_sync_log_retention.py`: TASK-38955's quit-time retirement and its retention-pass worker handles.
  - `Tests/RAG_Search/test_keyword_leg_db_resolution.py`: TASK-38956's FTS5 pool retirement.
  - `Tests/Web_Server/test_mobile_service.py`: TASK-38972's mobile shutdown.
  - `Tests/Library/test_library_ingest_runner.py`: TASK-38954's Research startup ordering. Three of its nodes run only in the label-triggered `task-603-platform-evidence.yml`.

**Local timings on #3435** (pytest time; 3 runs at `-n 4`, then 1 serial run with the dreams step's flags `-p no:xdist --storage-lease-guard --timeout=180`; load 17-30):

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
- [x] #1 `Tests/DB/test_sync_log_retention.py` runs whole on a PR lane, appended at the end of the dreams-boundary step (never ahead of the storage-pausing tests pinned first); the lane pins in `Tests/CI` (`DREAMS_BOUNDARY_TARGETS`, the step argv pin in `test_response_rules_critical_gate.py`) are updated, and three timings of the step's exact command are recorded
- [ ] #2 `Tests/Web_Server/test_mobile_service.py` runs on a PR lane once TASK-38992 makes it green, with three timings of that lane's command recorded
- [ ] #3 `Tests/RAG_Search/test_keyword_leg_db_resolution.py` runs on a PR lane; its evidence is a green CI run of that lane (locally every node hits TASK-38111), linked in the notes
- [ ] #4 `Tests/Library/test_library_ingest_runner.py` either leaves no live lease (0 lease-guard errors under `--storage-lease-guard`, with the before count of 130 recorded) and joins the guarded step, or the owner places it on a lane without the guard; the decision is recorded
- [ ] #5 Recent job times of every lane touched are quoted before and after; the fast PR contract is not used
<!-- AC:END -->

## Implementation Plan

1. Read the recent job and step times of the dreams-boundary lane (read-only `gh`).
2. Append `Tests/DB/test_sync_log_retention.py` to "Run Dreams and Guardian boundary contracts", after the last target and before `--storage-lease-guard`. Update `DREAMS_BOUNDARY_TARGETS` and the step argv pin in `Tests/CI/test_response_rules_critical_gate.py`.
3. Run the step's exact command 3 times (`jobstep.sh`) and record the times, with the base step for comparison.
4. Check the other three files against their criteria: mobile waits for TASK-38992; the keyword-leg and ingest-runner files must pass in the step, under the guard.

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**AC#1 (met).** `Tests/DB/test_sync_log_retention.py` is now the last target of "Run Dreams and Guardian boundary contracts" (`.github/workflows/derived-artifacts.yml`), after `Tests/UI/test_library_modal_dismissal.py` and before `--storage-lease-guard`. The storage-pausing test stays first. The file runs on the bootstrap profile (`Tests/DB` is in `keep_bootstrap_profile`). TASK-32873's note in `derived-artifacts.yml` keeps bootstrap-profile suites out of sandboxed invocations, because their enrollment can break the sandboxed suites that follow them. So the file has to stay after every sandboxed target in this step; today nothing follows it. A later append after it would need its own check. `DREAMS_BOUNDARY_TARGETS` (`Tests/CI/test_ci_queue_pressure_contract.py`) and the step argv pin (`Tests/CI/test_response_rules_critical_gate.py`) gained the same entry. The file had 36 tests when timed below (TASK-38955, TASK-39047, TASK-39048) and passed serially under the guard in 18.6 s. A 37th, the kernel no-op from review r11s-rev1, came later. Under load 45-58 the same file took 44-138 s, on base too: one ChaChaNotes or media fixture setup took about 50 s, and one setup hit `private_sqlite_helper_timeout`.

**Step timings (AC#1, AC#5 local).** Each run is the step's exact command, `jobstep.sh fu-r11-storage "Run Dreams and Guardian boundary contracts"` (serial, `--storage-lease-guard --timeout=180`), and every run was green. Other agents kept the 18-CPU machine at load 38-56.

| Run | Tree | Result | pytest time | Load (1 min) |
|---|---|---|---|---|
| 1 | this branch | 906 passed | 512.2 s | 43-56 |
| 2 | this branch, while the base run below ran | 906 passed | 724.6 s | 44-53 |
| 3 | this branch | 906 passed | 547.4 s | 38-50 |
| base | `fu-base-r11` (dev 2684e5453f), at the same time as run 2 | 870 passed | 757.6 s | 44-53 |

The added file is about 18 s of the step. Two runs here, and the base run, went past the brief's 12-minute mark only under this load. The base step without the file took longer (12:37) than run 2 with it (12:04).

**Lane times before (AC#5), from CI, read-only.** In the five most recent successful derived-artifacts PR runs, 42178132011, 42148997122, 42141155746, 42136873708 and 42136862171, the dreams-boundary job took 6m30s, 8m54s, 6m41s, 5m54s and 7m34s. This step took 3m57s, 2m14s, 2m32s and 3m06s in the last four of those runs (the first run's step times were not in its job record). After times need a CI run of this branch, so AC#5 stays open. `pr-fast-lane` is not used.

**AC#2 (open, blocked).** `Tests/Web_Server/test_mobile_service.py` has 32 passed and 1 failed at base and here. The failure is `test_serve_wiring_attaches_relay_only_when_broker_exists` (`AttributeError … revoke_browser_session`), which TASK-38992 owns, and TASK-38992 is still To Do. The criterion waits for it, so the file is not laned.

**AC#3 (NEEDS_CONTEXT).** All 11 nodes of `Tests/RAG_Search/test_keyword_leg_db_resolution.py` fail with `RecoveryRequired: raw_source_selection_changed`, at base and here. This is not a local-only trap, so the CI lane would be red as well:
- `Tests/conftest.py:181` imports `tldw_chatbook.Chunking` unconditionally at collection, and `Chunk_Lib.py:301` reads config there.
- So the config source is registered with the bootstrap profile's path in every process, CI included. A probe of `raw_participants._register_raw_participant` showed that first registration.
- Each node's `RAGConfig()` then reads config under its own per-test path, and `_register_raw_participant` refuses.
- Making the file pass means a bootstrap-profile enrollment or a config selection in its harness, which is TASK-38111's fix. BRIEF-RULES forbids other tasks from making it. So the file is not laned, and there is no green CI run to link.

**AC#4 (NEEDS_CONTEXT).** The lease count is unchanged here: serially under `--storage-lease-guard`, the file gave 86 failed, 70 passed, 1 skipped and 130 lease-guard errors. 128 of the errors are `library_ingest.db` on MainThread: `_make_db` builds MediaDatabases that the tests never close. The other four are one each of `workspace.sqlite` on asyncio_0 and on asyncio_1, `retry-workspaces-server.sqlite` and `retry-workspaces-local.sqlite`. 45 of the 126 leaking nodes pass. The leases look fixable: a file-local fixture could close every store the tests open (MediaDatabase now has `close_idle_connections`, TASK-39047). The file still could not join the guarded step:
- 85 of its 86 failures are the same TASK-38111 refusal (`_resolve_ingest_backend` -> `get_cli_setting`), and CI fails them the same way, as for AC#3.
- That includes two of the three nodes `task-603-platform-evidence.yml` runs, and that workflow's recent runs were all skipped.
- The 86th failure is `test_real_app_wires_research_association_and_restores_before_startup_resume` (`assert ['research_so...tion_startup'] == ['research_so...ging_startup']`), the "staging unavailable" path in TASK-38954's notes.

I did not fix the leases: that alone does not let the file join, and the leases of the 85 refused nodes cannot be checked until they run. There is no owner decision for a lane without the guard. One of these is needed: TASK-38111 lands first; this task is allowed to enroll the file (bootstrap profile, or a config selection in `_IngestRunnerHarness`); or the owner places the file on a lane without the guard.

**Files.** `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`, `Tests/CI/test_response_rules_critical_gate.py`. `Tests/CI` at `-n 4`: 1215 passed, 1 skipped.
<!-- SECTION:NOTES:END -->
