---
id: TASK-38978
title: "The real-key Ctrl+Q in-flight modal tests run on no PR lane"
status: To Do
created_date: 2026-11-06 01:42
labels:
- follow-up
- ci
- tests
- quit
priority: low
references:
- Tests/UI/test_app_quit_in_flight_modals.py
- scripts/ui_pr_gate_census.txt
- scripts/check_ui_pr_gate_census.py
- .github/workflows/derived-artifacts.yml
dependencies:
- TASK-38747
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down; checked on dev 29592ecc9b and on the census and workflows as they stand once the round-8 branches merge, 2026-11-05.

`Tests/UI/test_app_quit_in_flight_modals.py` holds the real-key Ctrl+Q tests for guarded modals (TASK-33622.15, TASK-38747). It is in no workflow and not in the census.

The census checker's comment (`scripts/check_ui_pr_gate_census.py`, about lines 100-104 after the merges) keeps the file out, saying it takes "~2.7 min" and stays off "until the lane has a shard with room". The measured time is far lower:

| Run | Result | Time | Load |
|---|---|---|---|
| Serial at `--timeout=180` (the UI Fast Lane configuration) | 13 passed | 47.8 s | 5.8 |
| `-n 4`, 3 runs | 13 of 13 each time | 25-29 s each | 11.9-12.5 |

TASK-38747 adds 3 tests to the file, making 16.

TASK-33622.15 and TASK-33622.17 (both Done) only noted that the file was off the lane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The file runs in the UI Fast Lane, and the census checker's comment no longer says it stays out
- [ ] #2 Before it is added, it passes 3 of 3 serial runs at --timeout=180 on dev with TASK-38747's tests, with the measured time recorded (47.8 s for 13 tests on dev)
- [ ] #3 The shard it joins still ends at least 5 minutes under its 20-minute cap in its slowest measured run
<!-- AC:END -->
