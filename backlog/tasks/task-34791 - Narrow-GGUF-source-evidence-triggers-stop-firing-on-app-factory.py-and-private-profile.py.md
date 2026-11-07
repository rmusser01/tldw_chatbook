---
id: TASK-34791
title: 'Narrow GGUF source evidence triggers: stop firing on app_factory.py and private_profile.py'
status: Done
created_date: 2026-10-11 03:29
updated_date: 2026-10-11 03:30
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The two-week review of CI throughput sub-project 1 (TASK-33160) measured the changed workflows at 70% below the runner-minutes baseline against an 80% target. The remainder is the two GGUF evidence workflows, and 110 of the source workflow's 230 runs in 13 days were fired only by two shared test-harness files. The owner chose to drop those two trigger paths from the source workflow and keep every other trigger.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 GGUF source evidence does not start for a pull request whose only matching files are Tests/UI/app_factory.py or Tests/private_profile.py
- [x] #2 pyproject.toml, Tests/conftest.py and Tests/UI/conftest.py still trigger both GGUF evidence workflows
- [x] #3 The coverage gap this adds is recorded in ADR-103
- [x] #4 The pinned trigger-path test matches the workflow
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Remove the two paths from the source workflow's pull_request path filter.
2. Update the pinned path list in Tests/CI/test_task2062_2_gguf_source_evidence.py.
3. Record the added gap in ADR-103's 2026-09-27 amendment.
4. Run the two GGUF workflow contract tests and preflight.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Removed `Tests/UI/app_factory.py` and `Tests/private_profile.py` from the `pull_request` path filter of `.github/workflows/task-2062-2-gguf-source-evidence.yml`. Nothing else changes: the import workflow is untouched, and `pyproject.toml`, `Tests/conftest.py` and `Tests/UI/conftest.py` still trigger both workflows.

**Why these two.** Attribution of every GGUF run from 2026-09-28T02:01Z to 2026-10-11T00:00Z (337 runs from 23 PRs, 4,309 runner-minutes) against each PR's file list: 110 of the source workflow's 230 runs, 1,706 of its 3,903 minutes, matched only these two paths. Replaying the window without them leaves 2,604 minutes, 202 a day, 82% below the ~1,100 baseline (target 80%). Dropping `pyproject.toml` and both conftests instead gives the same 82% but removes the only macOS and Windows install signal a dependency edit gets, which Qodo flagged on #2860; the owner chose this cut on 2026-10-11.

**Trade-off.** These paths were restored on #2860 after review, because the GGUF UI tests are not in the PR gate. An edit confined to either file no longer exercises the GGUF source screens on any PR. ADR-103's 2026-09-27 amendment records that.

**Evidence.** `Tests/CI/test_task2062_2_gguf_source_evidence.py` and `test_task2062_1_gguf_import_evidence.py`: 9 passed. The attribution uses each PR's current file list as an approximation of what the path filter saw at push time. The 82% is a replay of 13 days, not a measurement; TASK-33160's runner-minutes criterion stays open until it is re-measured on runs after this merges.

Files: the source workflow, its contract test, ADR-103, this task.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
