---
id: TASK-42628
title: Run the CI contracts on xdist workers in the light job
status: Done
assignee:
  - '@claude'
created_date: '2026-11-27 06:10'
updated_date: '2026-11-27 06:10'
labels:
  - ci
dependencies:
  - TASK-42521
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The light dev-PR job ("Dev smoke and affected contracts") runs `Tests/CI` in one serial pytest process before the PR's own tests. Locally that took about 10 minutes, most of a light run, and the job has a 30-minute cap. The owner agreed on 2026-11-27 to run those contracts on 4 workers so light PRs merge faster.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The light job runs `Tests/CI` on 4 xdist workers; every other selected file keeps its own serial process
- [x] #2 The light job's install step provides pytest-xdist
- [x] #3 `Tests/CI` passes on 4 workers on Linux with the light job's install set
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Pass `-n 4` to the `Tests/CI` group in `scripts/ci/run_light.py`.
2. Add pytest-xdist to the dev-light install line.
3. Pin both in `Tests/CI/test_tiered_ci.py`.
4. Run `run_light.py` end to end locally and in an ubuntu:24.04 container.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`run_light.py` adds `CONTRACT_WORKERS` (`-n 4`) to the `Tests/CI` group only. A bootstrap_profile file still never shares a process with a sandboxed one, because each selected file keeps its own process. `Tests/CI` already ran in one shared process, so splitting it over 4 workers mixes nothing new. The dev-light install line gains `pytest-xdist`.

Tests: `test_light_job_installs_the_workers_its_runner_asks_for` pins the install. The runner test now checks that only `Tests/CI` gets `-n 4`.

Evidence (macOS, Python 3.12.11, scratch HOME): `run_light.py` with the three base smoke files took 145 s end to end (`Tests/CI`: 2010 passed, 2 skipped in 81 s). The serial run was 817 s on 2026-11-24.

This PR runs the full CI itself (`scripts/ci/`, `.github/`), so its own run does not exercise dev-light. The Linux evidence comes from an ubuntu:24.04 container (Python 3.12.3, SQLite 3.45.1, user `runner` uid 1001, ext4 volumes for HOME and `/tmp`, the dev-light install line, the tree committed into a fresh git repo). There, `run_light.py` with the three smoke files exited 0 in 118 s, with `Tests/CI` on 4 workers at 2009 passed, 3 skipped in 81 s.

A first container run had one red: `test_task598_external_parakeet_evidence.py::test_supervisor_terminates_a_descendant_in_a_new_session`. It failed serially there too, 3 of 3, so xdist is not the cause. The container's PID 1 was `sleep`, which never reaps a killed descendant, so its process group still looked alive. Under `docker run --init` it passes (52 passed). GitHub's runners have a reaping init.

Files: `scripts/ci/run_light.py`, `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_tiered_ci.py`.
<!-- SECTION:NOTES:END -->
