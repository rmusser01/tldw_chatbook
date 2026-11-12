---
id: TASK-35882
title: >-
  Re-check UI Fast Lane shard minutes once CI has timed TASK-35660's two new
  resend pilots
status: To Do
assignee: []
created_date: '2026-10-21 00:35'
updated_date: '2026-10-22 18:02'
labels:
  - tests
  - ci
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-35660 adds two app-booting pilots to Tests/UI/test_console_turn_resend_ui.py, which is in scripts/ui_pr_gate_census.txt and so runs in the UI Fast Lane shards (job limits 20 and 35 minutes). No CI timing for them exists yet. Locally each took about 4-5 s at host load ~21 and 35-45 s at load ~100, plus teardown. scripts/ui_lane_shard_replay.py scores each census file at the worst of the CI runs it is given, so the check needs the shard logs of real runs that include the two tests. Raised by TASK-35660's independent review on 2026-10-20.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 scripts/ui_lane_shard_replay.py, run over the shard logs of at least two CI runs that include the two new pilots, is recorded in the notes with each shard's minutes
- [ ] #2 Every UI Fast Lane shard stays under its job limit with the margin the census comments ask for, or the census tail is re-ordered with --write and the replay re-run until it does
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Replay with CI timing of the two pilots (2026-10-22, on dev 7700c63a68's 13-shard census).** PR #3190's run 39610893391, attempts 2 and 3 (ten-shard deal, both include the two pilots), ran Tests/UI/test_console_turn_resend_ui.py in job 1: 17 of 17 in 75.7 s and 76.6 s. Neither pilot is in that job's 25 slowest durations (the 25th were 10.45 s and 10.58 s), so each call phase took less. Dev's own 15-test file took 75.0-100.2 s in four green thirteen-shard PR runs (39799596144, 39744284533, 39734506492, 39730705223).
- `scripts/ui_lane_shard_replay.py --shards 13` over those four dev runs: 13.8 15.8 16.8 16.2 15.5 15.8 13.8 15.8 14.1 14.7 13.7 14.8 12.8, max 16.8 pytest minutes.
- Over the same four plus PR #3190's attempts 2 and 3: 13.9 16.3 17.4 16.6 15.6 15.8 13.9 15.9 14.4 15.1 14.0 14.8 14.3, max 17.4. Job 6 holds the resend file in the 13-shard deal and stays at 15.8: the replay scores the file at dev's worst run, which is slower than either PR run. The other jobs rise because the older ten-shard logs score some unrelated entries higher.
- AC #2 is not met, but not because of the pilots. Dev's own census leaves under 3 minutes in three of those four runs (job wall max 18.1, 17.4, 17.5, 16.7 min), and the replay of dev alone is already past TASK-35834's 1,000 s projection. Filed as TASK-36140; re-check AC #2 once it lands.
<!-- SECTION:NOTES:END -->
