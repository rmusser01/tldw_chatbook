---
id: TASK-38978
title: "The real-key Ctrl+Q in-flight modal tests run on no PR lane"
status: Done
assignee: ['@claude']
created_date: 2026-11-06 01:42
updated_date: '2026-11-06 17:26'
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
- [x] #1 The file runs in the UI Fast Lane, and the census checker's comment no longer says it stays out
- [x] #2 Before it is added, it passes 3 of 3 serial runs at --timeout=180 on dev with TASK-38747's tests, with the measured time recorded (47.8 s for 13 tests on dev)
- [x] #3 The shard it joins still ends at least 5 minutes under its 20-minute cap in its slowest measured run
<!-- AC:END -->

## Implementation Plan

1. Qualify before adding: `Tests/UI/test_app_quit_in_flight_modals.py` 3 serial runs at `--timeout=180` (the UI Fast Lane's flags) on dev (the read-only base worktree, 16 tests) and on this branch (17 with TASK-38980's promote test), with times and load.
2. Read recent UI Fast Lane shard times (`gh run view --json jobs`, read-only) and pick the tail slot by the round-robin deal (`entries[index::32]`); record the slowest run of the shard it joins plus the file's measured time, which must stay at least 5 minutes under the 20-minute cap.
3. Append the file as one whole-file entry at the census tail, set `MINIMUM_FILES` to the new count with a dated comment, and rewrite the checker's "stay out until the lane has a shard with room" comment to say where the file runs now.

## Implementation Notes

**Approach.** `Tests/UI/test_app_quit_in_flight_modals.py` joins the UI census as one whole-file entry at the tail, `MINIMUM_FILES` goes to the new entry count (1751 on this base, with TASK-38979's file and TASK-38980's node appended beside it), and the checker's TASK-33622.15 comment now says the file runs in the lane (`test_console_video_picker_cancel.py` is still out). No workflow file changed: the UI Fast Lane reads the census.

**Qualification (AC#2), `pytest -p no:xdist --timeout=180`, serial.**
- Base, the read-only fu-base-r11 worktree (16 tests, with TASK-38747's three): 3 of 3, 115.2 s, 125.0 s and 118.6 s at load averages 16.0-30.2.
- This branch (17 tests, with TASK-38980's Promote test): 3 of 3, 135.0 s, 118.4 s and 126.6 s at load 26.9-47.5.
- The filing measured 47.8 s for 13 tests at load 5.8. The machine was shared with other agents this round, and local wall time scales with load.

**Shard (AC#3).** By the census's round-robin deal (`entries[index::32]`) the file is shard 21 of 32 on this base. That shard's slowest job over the 7 completed shard runs read from 8 recent PR runs (42150847285 to 42191565464, read-only `gh run view --json jobs`) was 13.5 min. One census file measured both ways ran 12.1 s in a CI shard log and 10.5 s here at load 31, so CI runs at about the speed of a moderately loaded local run. At the filing's 47.8 s for 13 tests (about 1.1 min for 17), the shard ends near 14.6 min, 5.4 min under the 20-minute cap. The whole shard, run locally serially with the lane's flags, passed: 223 passed in 892 s at load 32. Other round-11 branches also append at the tail, so a rebase moves the file to another shard, which the coordinator should re-check.

**Files.** `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`.
