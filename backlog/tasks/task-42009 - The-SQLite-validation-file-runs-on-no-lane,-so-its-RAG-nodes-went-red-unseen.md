---
id: TASK-42009
title: "The SQLite validation file runs on no lane, so its RAG nodes went red unseen"
status: Done
assignee:
- '@claude'
created_date: 2026-11-19 19:39
updated_date: 2026-11-20 16:50
labels:
- follow-up
- tests
- backup
priority: low
references:
- Tests/Backup_Recovery/test_sqlite_validation.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41718 (round 17). `test_sqlite_validation.py`'s two `test_native_rag_indexing_schema_zero_survives_restricted_validation` nodes were red on dev until TASK-41718 fixed the RAG schema check; nothing noticed because the file runs on no lane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The file runs on a PR or dev-push lane, with its time recorded
<!-- AC:END -->

## Implementation Plan

1. Run the file alone at base to confirm it is green and sandboxed.
2. Pick a lane with headroom that runs a plain process (architecture-guards: 6-11 of 20 min on recent PR runs) and add the file before its bootstrap entries.
3. Update the Tests/CI pin (ARCHITECTURE_GUARD_TARGETS) and the step note; record the serial time with the lane's flags and the load.
4. Run Tests/CI and the whole lane step locally.

## Implementation Notes

Approach: the file was green at base (TASK-41718 fixed its RAG nodes); it needed a lane. Time with the lane's flags (`-p no:xdist --timeout=180 --tb=short`): 82 passed in 55.5 s, load 31. A PR lane rather than the dev-push `backup-owner-suites-dev`, so a PR that breaks it is stopped before the merge.

Lane: `architecture-guards` ("Architecture guard contracts"), a plain serial process on every PR. Over 21 recent PR runs (to 2026-11-20) the job took 6-11 min of its 20-min cap, the most headroom of the plain-process jobs read (recovered-suites 14-25 of 30, architecture-review 9-15 of 20, admission 4 up to 27 of 35). The new targets are sandboxed, so they sit after `test_restore_plan_excluded_directories.py` and before the ChaChaNotesDB files and the last (bootstrap_profile) node (TASK-32873); none imports hypothesis, pytest-mock or xdist, which the lane's install line lacks (import check with those modules blocked).

Verification: Tests/CI at -n 4: 1811 passed, 2 skipped (`ARCHITECTURE_GUARD_TARGETS` updated; the step-shape and disjoint-target pins pass). The whole step locally, serially with its flags (46 targets): 748 passed, 1 skipped in 421 s at load 20-22.

Files: `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`.
