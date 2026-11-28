---
id: TASK-41519
title: "The macOS plugin qualification job runs 20 to 29 minutes of its 30 minute cap, and the per-child caps are not tightened"
status: In Progress
assignee:
- '@claude'
created_date: 2026-11-17 05:27
updated_date: 2026-11-28 12:15
labels:
- follow-up
- ci
- plugins
- needs-owner-decision
priority: medium
references:
- .github/workflows/derived-artifacts.yml
- .github/workflows/plugin-qualification.yml
- Tests/Plugins
- backlog/docs/ci-lane-headroom-2026-11-28.md
dependencies:
- TASK-41104
- TASK-41217
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by plgb (round 15, TASK-41104 and TASK-41217). The macOS plugin-qualification job takes 20.0 to 28.6 of its 30 minutes, so a CPU-busy hang could hit the job cap before any per-child cap. TASK-41104's per-child caps should be tightened once the PR's qualification run records per-child CI times.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Per-child CI times from a qualification run are recorded in the notes
- [ ] #2 Per-child caps are set from them, and a hang fails on its child cap before the job cap
- [ ] #3 The job's typical time has at least 5 minutes of margin
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Check that nobody is on it: the board on every ref, open pull requests and remote refs that edit the workflow, dev's history of the cap.
2. Read every macOS job since 2026-11-26 from the API with the annotation of each cancellation, and a sample of their logs: where the minutes go, and whether a cancelled job had finished its tests.
3. Cost the ways out: a higher cap, fewer files on the macOS leg, per-child caps.
4. Put the options to the owner. A cap raise has been the owner's decision every time (ADR-103).
5. Apply the choice, then measure 20 pull-request runs (AC #3).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-11-28, re-measured. Plan steps 1 to 4 are done; step 5 waits for the owner. AC #1 and AC #2 (per-child times and caps) are TASK-41104's follow-through and are not touched here; the numbers for AC #3 are below. The pull request that carries this note also applies TASK-42779's move in `derived-artifacts.yml`, so its CI is the full run (its preflight chose the full tier on a411112e1c). It changes neither this job nor `plugin-qualification.yml`, and nothing in this task is applied.

**Which job.** `qualify (macos-latest)` of `.github/workflows/plugin-qualification.yml` (`timeout-minutes: 30`, a three-OS matrix, `max-parallel: 3`), which runs on pull requests that touch the plugin paths. It is not a required check: the only required context on `dev` is "Derived artifacts reproduce from their sources" (strict, `enforce_admins`), so a cap hit costs a runner and a rerun, not a merge. The repository is public, so hosted minutes are free. No `Tests/CI` test pins the 30 (`test_plugin_qualification_workflow.py` does not mention it).

**Awaiting the owner.** Which option below. The recommendation is option 2, cap 40. A cap raise is the owner's decision, so it is not applied.

**Checked first.** This task is the only one on any ref about this job's cap (TASK-41104 and TASK-41217, which it depends on, are Done). Of the remote refs, five edit `plugin-qualification.yml` against their merge base with `origin/dev`, all by adding path filters or lines (no deletions); none touches the cap or the pytest step. No open pull request changes the cap.

**Measured.** GitHub API job records of the 57 `plugin-qualification.yml` runs created from 2026-11-26 to 2026-11-28, each job execution counted once, and 7 macOS logs (the fastest, the quartiles, the slowest and the two cancellations). 57 macOS jobs: 55 finished (median 25.0, p90 28.7, max 29.6 minutes; 7 took 28 or more, 4 took 29 or more) and 2 were cancelled at 30 minutes. A cancellation is "at the cap" only when its annotation says so: both read "The job has exceeded the maximum execution time of 30m0s". Both are PR 3345's branch (job 140440507761, run started 2026-11-27 10:13Z; job 140711431149, 20:15Z), and neither was hung. Each printed pytest's summary at the 30-minute mark (824 passed and 1 failed in 28:09; 840 passed in 28:18) and was cut 1 to 2 seconds later, so the job needs about 30.1 minutes there.

The job is one pytest step ("Exercise explicit lifecycle and regression boundaries": about 834 tests of `Tests/Plugins`, the hook v2 files, the startup performance file and the private-SQLite inventory) of median 23.4 and max 28.1 minutes over the 55 finished jobs; the other steps together take about 1.7 minutes (checkout 0.7, setup-python 0.2, install 0.4). In the 7 logs the same tests ran 17.4 to 28.3 minutes of pytest time (1,045 to 1,699 s), all on `MacCloud` region `westus`: a spread of 1.63 times, with the runner-capacity notice ("jobs targeting macOS arm64 runners may experience longer queue times") on the cancelled jobs. The median log's 10 slowest tests are 454 s of its 1,432 s (`test_recovery.py::test_plugin_backup_restores_in_place_and_waits_for_authority_recovery` 99.9 s, three `test_private_sqlite_inventory.py` census tests 125 s, `test_git_acquisition.py::test_https_host_helpers_tls_and_exact_origin` 80 s, `test_source_service.py::test_selected_remote_entry_requires_exact_displayed_origin_before_network` 41 s, `test_owned_hook_acceptance.py::test_installed_hook_aggregate_admission_and_teardown` 39.5 s); no file dominates. Per-file times cannot be read from these logs (the progress lines carry no file names).

AC #3 asks that the job's typical time have at least 5 minutes of margin. Over all 57 jobs, the two cut ones at 30.1, the median is 25.3, which leaves 4.7 under cap 30; over the 55 finished jobs alone it is 25.0 and leaves exactly 5.0. Over all 57 the p90 leaves 1.0, and the slowest finished job left 0.4.

**Options.** Observed over the 57 jobs, the two cut jobs at the 30.1 minutes they needed; no runner speed model (the logs have no per-file times):

| | Change | Runners | Median | p90 | Slowest | Under the cap | Over the cap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | none (cap 30) | 0 | 25.3 | 29.0 | 30.1 | -0.1 | 2 of 57 |
| 1 | cap 30 to 35 | 0 | 25.3 | 29.0 | 30.1 | 4.9 | 0 |
| 2 | cap 30 to 40 | 0 | 25.3 | 29.0 | 30.1 | 9.9 | 0 |

1. **Cap 35.** Meets AC #3 (9.7 minutes over the 57-job median) and leaves 4.9 on the slowest job seen. The macOS spread is 1.63 times on identical tests, so a slower draw than any seen would use that up.
2. **Cap 40.** One line in the workflow. No runner, no required check at risk, nothing pinned. A hung job holds its macOS runner 10 minutes longer, which the per-child caps of AC #2 are meant to prevent. Stays below the 45-minute busy-job deadline of the self-hosted controller (ADR-359), which does not apply to hosted macOS.
3. **Fewer files on the macOS leg** (for example the private-SQLite inventory's 125 s, which also runs on Linux in `pr-fast-lane-arch`) is not costed: the matrix shares one pytest list, and what the macOS leg qualifies ("only executed passing boundaries qualify") would change.

**Recommendation: option 2, for the room it leaves a slow draw. AC #3 does not separate the two caps: both meet it.** The two hits finished their tests, so it is a slow runner draw and not a hang, the lane is not required, and a cap is the smallest change. If the owner would rather bound a hung job sooner, option 1 also meets AC #3. AC #2 stays open and should be done first if the owner would rather keep the cap and shorten a hang.

**How it was measured.** Job records, annotations and step times from the GitHub API; pytest's own summary and slowest-25 tables from the logs. The scripts and per-job tables are not in the repository.
<!-- SECTION:NOTES:END -->
