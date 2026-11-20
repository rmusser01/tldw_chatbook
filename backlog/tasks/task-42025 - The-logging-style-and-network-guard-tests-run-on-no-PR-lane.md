---
id: TASK-42025
title: "The logging style and network guard tests run on no PR lane"
status: Done
assignee:
- '@claude'
created_date: 2026-11-19 19:39
updated_date: 2026-11-20 16:50
labels:
- follow-up
- tests
- ci
priority: low
references:
- Tests/test_stdlib_logging_format_style.py
- Tests/test_network_guard.py
- .github/workflows/derived-artifacts.yml
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41708 and TASK-41714 (round 17). `Tests/test_stdlib_logging_format_style.py` (the loguru `%s`-style and stdlib brace-style scans) and `Tests/test_network_guard.py` (including the new DNS refusal) run on no PR lane, so a regression in either guard would land unseen.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both files run on a PR lane, with their times recorded and the CI pins updated
<!-- AC:END -->

## Implementation Plan

1. Run both files alone at base with the lane's flags.
2. Add both to architecture-guards (a plain process; it already holds the diagnostic and log-sanitizer guards), before its bootstrap entries.
3. Update ARCHITECTURE_GUARD_TARGETS and the step note with the measured times and load.
4. Run Tests/CI and the whole lane step locally.

## Implementation Notes

Approach: both files were green at base and need no bootstrap profile; they join the lane that already holds the persistent-diagnostic boundary and log-sanitizer guards. Times with the lane's flags (`-p no:xdist --timeout=180 --tb=short`): `test_stdlib_logging_format_style.py` 10 passed in 6.2 s (load 28); `test_network_guard.py` 30 passed, 1 skipped (the Windows-only socketpair case) in 1.4 s (load 26). The network guard's DNS tests patch `_socket.getaddrinfo`, and its opted-in test connects to a listener it owns, so the lane resolves no real name. CI pins updated: `ARCHITECTURE_GUARD_TARGETS`.

Lane: `architecture-guards` ("Architecture guard contracts"), a plain serial process on every PR. Over 21 recent PR runs (to 2026-11-20) the job took 6-11 min of its 20-min cap, the most headroom of the plain-process jobs read (recovered-suites 14-25 of 30, architecture-review 9-15 of 20, admission 4 up to 27 of 35). The new targets are sandboxed, so they sit after `test_restore_plan_excluded_directories.py` and before the ChaChaNotesDB files and the last (bootstrap_profile) node (TASK-32873); none imports hypothesis, pytest-mock or xdist, which the lane's install line lacks (import check with those modules blocked).

Verification: Tests/CI at -n 4: 1811 passed, 2 skipped (`ARCHITECTURE_GUARD_TARGETS` updated; the step-shape and disjoint-target pins pass). The whole step locally, serially with its flags (46 targets): 748 passed, 1 skipped in 421 s at load 20-22.

Files: `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`.
