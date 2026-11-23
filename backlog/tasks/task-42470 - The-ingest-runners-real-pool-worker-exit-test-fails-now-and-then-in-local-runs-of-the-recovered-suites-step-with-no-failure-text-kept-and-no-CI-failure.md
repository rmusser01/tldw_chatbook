---
id: TASK-42470
title: "The ingest runner's real-pool worker-exit test fails now and then in local runs of the recovered-suites step, with no failure text kept and no CI failure"
status: To Do
assignee: []
created_date: 2026-11-23 06:38
updated_date: 2026-11-23 07:09
labels:
- follow-up
- library
- ingest
- tests
- flaky
priority: medium
references:
- Tests/Library/test_library_ingest_runner.py
- .github/workflows/derived-artifacts.yml
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 22 of the follow-up burn-down (2026-11-23), by TASK-42426's and TASK-42412's implementers. Checked on the round-22 stack 336d15338b (dev 2f7790873a plus round 22's stacked branches); paths start at the repository root.

**Why.** `Tests/Library/test_library_ingest_runner.py::test_real_pool_worker_exit_is_reported_for_owning_generation` (line 1796) runs in the required "PR lane (recovered suites)" job's "Run the recovered suites" step (`.github/workflows/derived-artifacts.yml`, where the file's entry is line 1856; flags `-p no:xdist --storage-lease-guard --timeout=180`). The test kills a real parse-pool worker (`pool.apply_async(os._exit, (17,))`, line 1841) and expects the job to fail as not permanent, with the pool and its generation released (lines 1843-1852). It failed in 2 of about 5 local runs of the whole step with CI's flags. TASK-42426 recorded it failing at base with the same flags (2004 passed, 2 failed, in 24:41 at load 34-46), and TASK-42412's second step run failed it too (load 24-46). It passed in the other runs. Neither implementer kept the failure text: TASK-42412's step script kept only the last 15 lines.

**CI.** In a scan of every attempt of the Derived Artifacts runs created from 2026-11-19 06:00Z to 2026-11-23 06:00Z (321 runs; the first page of run 45080343651's job list returned Server Error in the scan and, re-read, holds one more passing attempt), the recovered-suites job completed at least 168 times (129 passed, 39 failed). This node failed in none of the 38 failed jobs whose logs were read, and it passed in every passing one. The failures there were the managed received acceptance test (TASK-42412), the stock skill-trust dispose cases (TASK-42426) and two other nodes. Alone, with CI's flags, it passed 5 of 5 at load about 51. So it is a local red under a loaded whole-step run, not yet a CI red; priority medium for that reason. Reproducing it may need the whole step, for example because other tests' pools or storage leases are still live when this worker exits; that is a guess.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Whole-step runs with CI's flags (or runs under a forced delay or load) keep their full output, and any failure of this test is recorded in the notes with its failure text
- [ ] #2 If it fails, its cause is named with evidence and fixed where it lives (the ingest runner's pool handling, the test, or another test in the step that leaves state behind); the test keeps its assertions and then passes in repeated whole-step or loaded runs with CI's flags (at least 10, or a soak recorded in the notes)
- [ ] #3 If at least 10 whole-step runs with CI's flags show no failure, the task closes with those runs and their loads recorded
<!-- AC:END -->
