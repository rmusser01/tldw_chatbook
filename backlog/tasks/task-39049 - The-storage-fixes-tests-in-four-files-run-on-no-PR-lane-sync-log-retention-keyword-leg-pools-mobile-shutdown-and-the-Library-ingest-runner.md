---
id: TASK-39049
title: "The storage fixes' tests in four files run on no PR lane: sync-log retention, keyword-leg pools, mobile shutdown and the Library ingest runner"
status: Done
assignee: ['@claude']
created_date: 2026-11-06 10:09
updated_date: '2026-11-12 19:40'
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
- [x] #2 `Tests/Web_Server/test_mobile_service.py` runs on a PR lane once TASK-38992 makes it green, with three timings of that lane's command recorded
- [x] #3 `Tests/RAG_Search/test_keyword_leg_db_resolution.py` runs on a PR lane; its evidence is a green CI run of that lane (locally every node hits TASK-38111), linked in the notes
- [x] #4 `Tests/Library/test_library_ingest_runner.py` either leaves no live lease (0 lease-guard errors under `--storage-lease-guard`, with the before count of 130 recorded) and joins the guarded step, or the owner places it on a lane without the guard; the decision is recorded
- [x] #5 Recent job times of every lane touched are quoted before and after; the fast PR contract is not used
<!-- AC:END -->

## Implementation Plan

1. Read the recent job and step times of the dreams-boundary lane (read-only `gh`).
2. Append `Tests/DB/test_sync_log_retention.py` to "Run Dreams and Guardian boundary contracts", after the last target and before `--storage-lease-guard`. Update `DREAMS_BOUNDARY_TARGETS` and the step argv pin in `Tests/CI/test_response_rules_critical_gate.py`.
3. Run the step's exact command 3 times (`jobstep.sh`) and record the times, with the base step for comparison.
4. Check the other three files against their criteria: mobile waits for TASK-38992; the keyword-leg and ingest-runner files must pass in the step, under the guard.
5. Round 12 (TASK-38111 landed): close the ingest runner's live leases in its own test code, then lane it and the keyword-leg pool file in the owner's new `recovered-suites` job, beside TASK-39142's keyword-leg files; time it 3 times with `jobstep.sh`.

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

**Lane times before (AC#5), from CI, read-only (review r11s-rev2's numbers).** On dev 2684e5453f (28 targets), across five runs (42267936447, 42263533424, 42253456601, 42252461355, 42247976071), the step took 409-485 s and the dreams-boundary job 11:02-13:07. `Tests/UI/test_library_modal_dismissal.py` alone took 155-181 s of that. The first version of these notes quoted job 5m54s-8m54s and step 2m14s-3m57s. Those runs predate the Library modal file and are stale. After times need a CI run of this branch, so AC#5 stays open. `pr-fast-lane` is not used.

**AC#2 (met, after review r11s-rev1).** TASK-38992 fixed the stand-in, so `Tests/Web_Server/test_mobile_service.py` is now laned too. It runs just before `Tests/DB/test_sync_log_retention.py`; the pins name both. The file is bootstrap-profile (module `pytestmark = pytest.mark.bootstrap_profile`) and guard-clean. It sits after every sandboxed target by TASK-32873's rule: the last three targets, the Library modal inventory, this file and the sync-log file, are all bootstrap-profile. Under the guard, serially at load 74-85 it gave 33 passed, 1 relay timeout and no lease finding. Its relay tests wait 5 s for events, so above load about 50 one or two of them time out, on base too. All three step runs below were green.

| Run | Result | pytest time | Load (1 min) |
|---|---|---|---|
| 1 | 941 passed, no guard finding | 1691.6 s | 67-108 |
| 2 | 941 passed, no guard finding | 1399.8 s | 46-66 |
| 3 | 941 passed, no guard finding | 1483.7 s | 59-82 |

Every run is far past the brief's 12-minute mark, because the machine ran at load 46-108 on 18 CPUs; runs 2 and 3 came right after a reboot. These runs cannot tell what the additions cost. The earlier numbers can:
- the step without these files took 512-547 s at load 38-56;
- the mobile file took 76 s serially at load 17-30 on #3435 (312 s at load 74-85 here);
- the sync-log file takes about 18 s.

So at moderate load the step should be about 10-11 minutes: close to the 12-minute mark, but under it. Projected for CI (review r11s-rev2): mobile adds about 61-135 s and sync-log about 19-34 s. With the Library route cycle another stream adds (about 50 s), the step comes to about 540-705 s and the job to about 13.2-16.8 min of its 20. Mobile is the file to move or split if margin is needed. Coordinator decision (2026-11-06): mobile stays on the step for now (one commit, bb138440ad).

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

**Files.** `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`, `Tests/CI/test_response_rules_critical_gate.py`. `Tests/CI` at `-n 4`: 1216 passed, 1 skipped (after the review round).
**Round 12 (2026-11-07).** TASK-38111 landed, so both remaining files run.

**Owner decision (2026-11-06):** "New CI job: add one derived-artifacts job, 'PR lane (recovered suites)', holding all of these files serially under --storage-lease-guard; one more runner per PR; ADR-103's job and runner counts updated." `Tests/RAG_Search/test_keyword_leg_db_resolution.py` and `Tests/Library/test_library_ingest_runner.py` run there, beside TASK-39142's four keyword-leg files, under `--storage-lease-guard`.

**AC#4 (met).** Before: 130 lease-guard errors (128 `library_ingest.db` on MainThread from `_make_db`; `workspace.sqlite` on asyncio_0 and asyncio_1; `retry-workspaces-local.sqlite` and `retry-workspaces-server.sqlite` on asyncio_0). The file's autouse fixture now closes every `_make_db` store on all threads (`close_idle_connections()`). The four WorkspaceDB handles were opened by research retries on the loop's executor threads, and SQLite refuses to close them from the test thread. So the fixture keeps a closer for each handle in a test-owned `threading.local()` of the thread that opened it, and the two research tests end with `shutdown_default_executor()`: each thread closes its own handle as it exits. After: 0 lease findings, 154 passed, 1 skipped, 2 failed. The 2 are TASK-39140's stale `error_detail` pin (fixed on fu/r12-ctlr, 489e2338f5). The keyword-leg file's FTS5 pool leaks are closed in `Tests/RAG_Search/conftest.py` (TASK-39142's notes). A lessons entry records the executor-thread trap.

**AC#3 (laned; CI evidence open).** Locally the keyword-leg pool file passes in the step, 11 of 11, guard-clean. The criterion asks for a green CI run of the lane, which needs a PR run of this branch, so it stays open.

**AC#5 (before quoted; after open).**

**CI before (read-only `gh`).** Dreams boundary job over 15 PR runs, 2026-11-06 14:36Z to 2026-11-07 00:55Z (42235403758 to 42321107301): job 9.80-14.05 min, step 6.10-8.93 min. None of them has round 11's three Dreams additions; no PR run of round-11 dev had finished when these were read (the queue held 04:33Z-05:31Z runs, all queued or cancelled). The recovered-suites job is new, so it has no before time.

**The recovered-suites step**, `jobstep.sh fu-r12-lanes "Run the recovered suites"` (14 targets, serial, `--storage-lease-guard --timeout=180`), 3 runs on 2026-11-07: 534.8 s (load 40-63), 703.2 s (load 40-60), 641.7 s (load 50-60). Each run: 723 passed, 1 skipped, 2 failed. The 2 are `test_executor_failure_uses_stable_job_terminal[artifact_incompatible-failed]` and `[engine_crashed-failed]` in the ingest runner, TASK-39140's stale `error_detail` pin, red on dev too and fixed on round-12 branch fu/r12-ctlr (489e2338f5); the step is green once both land. No lease finding in any run. The slowest run plus about 1.5 min of setup is about 13.2 min, 6.8 min under the job's 20-min cap; CI ran comparable lanes at 0.5-1.0 times this machine's loaded time.

**The Dreams step**, `jobstep.sh fu-r12-lanes "Run Dreams and Guardian boundary contracts"` (33 targets), 3 runs on 2026-11-07: 548.1 s (load 52 falling to 12), 422.4 s (load 12-8), 372.6 s (load 8-5); 1022 passed each, no lease finding. A diagnostic run of the same command with full output, at load 49-67, took 1214 s, also all green. The two TASK-39043 files take about 30 s of it serially (13 s and 17 s on #3434).

CI after times need a PR run of this branch.

**Files (round 12).** `Tests/Library/test_library_ingest_runner.py`, `Tests/RAG_Search/conftest.py`, `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`, `Tests/CI/test_derived_artifacts_workflow.py`, ADR-103.
**Review r12l-rev1 (2026-11-08).** The two bootstrap_profile Tests/UI files left the recovered-suites step for the UI census (the census does hold bootstrap_profile files: 512 entries already, and TASK-36478 restores the shared bootstrap config after each test; my earlier "may not hold" was wrong). TASK-39124's built-in pixel buddy file joined the step last. The final step, 13 targets, `jobstep.sh fu-r12-lanes "Run the recovered suites"`, 3 runs: 363.3 s (load 38 falling to 21), 742.9 s (load 21-50), 467.1 s (load 50-43); each 715 passed, 1 skipped, 2 failed (TASK-39140's pin, fixed on fu/r12-ctlr), no lease finding. The slowest run plus about 1.5 min of setup is about 13.9 min, 6.1 under the cap. Review r12l-rev1 also found that `_make_db`'s store list stayed populated all session when `Tests/integration/test_library_ingest_flow.py` imports `_make_db`; the list now exists only while this module's closing fixture runs.
**After the round-12 restack (2026-11-08).** The step now has 20 targets (TASK-39124's pixel buddy file, TASK-39119's six reply-pin nodes and the goal recovery file joined it). `jobstep.sh fu-r12-lanes "Run the recovered suites"`, 3 runs: 534.7 s (load 22-33), 353.8 s (load 22-28), 312.5 s (load 20-28); each 733 passed, 1 skipped, nothing red (TASK-39140's pin is fixed in the stack) and no lease finding. The slowest run plus about 1.5 min of setup is about 10.4 of the job's 20 min.
**AC#3, green CI run (round-13 closeout, 2026-11-12).** `Tests/RAG_Search/test_keyword_leg_db_resolution.py` ran 11 of 11 green in the `PR lane (recovered suites)` job of PR run 43248233114 (job 131298355582, 2026-11-12, branch codex/core-uat-fixes): the log line reads `Tests/RAG_Search/test_keyword_leg_db_resolution.py ........... [ 66%]`, and the step ends `733 passed, 1 skipped in 376.00s`. `Tests/Library/test_library_ingest_runner.py` passed in the same job. Of the 38 recovered-suites jobs in PR runs after round 12 merged, 37 were green; the red one (run 43085804799) was its branch's own change, and that branch's later runs passed.

**AC#5, after times (read-only `gh`, derived-artifacts.yml PR runs; the before times are quoted above).**
- **Dreams and Guardian boundary contracts** (this task's round-11 targets). After round 11 and before round 12, 72 PR runs (2026-11-08 15:04Z to 2026-11-10 15:11Z): job 9.87-16.75 min (median 13.15), step "Run Dreams and Guardian boundary contracts" 6.68-11.55 min (median 9.02). After round 12 merged, 38 PR runs (to 2026-11-12 17:03Z): job 11.07-16.17 min (median about 14.0), step 7.75-11.30 min (median about 9.9). The slowest job leaves about 3.3 min of the 20-min cap. The step itself failed once (run 43085804799, its branch's own change). Two other runs failed the job's separate admission-sensitive step on one Watchlists test (`test_a_rail_reopened_after_a_discarded_build_shows_the_landed_tree`, `NoMatches '#wl-tree'`), which is not this task's; TASK-40333 owns it.
- **PR lane (recovered suites)** (this task's round-12 targets). New, so there is no before. After: 38 PR runs, job 5.08-7.78 min, step 4.38-6.88 min.
- The fast PR contract was not used.
<!-- SECTION:NOTES:END -->
