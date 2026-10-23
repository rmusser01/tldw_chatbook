---
id: TASK-36512
title: UI Fast Lane shards 3 and 11 time out at the 20-minute cap on PR runs
status: To Do
assignee: []
created_date: 2026-10-23 22:46
labels:
- follow-up
- ci
- tests
priority: high
references:
- scripts/ui_pr_gate_census.txt
- scripts/ui_lane_shard_replay.py
- scripts/check_ui_pr_gate_census.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while burning down the Settings follow-ups (2026-10-23, dev 5b5920b8e2). UI Fast Lane jobs are being cancelled at their 20-minute cap with no failing test, which fails the required check. That has happened on:
- #3221, the merge queue's front PR;
- #3224 (twice);
- #3225 (shards 3 and 11).

Shard 3 took 13.2 and 17.5 pytest minutes on two green PR runs (40007935895, 40003892794), and had run 18.6 when it was cancelled on 40024354185. Two whole-file entries are 45-50% of it:
- `Tests/UI/test_settings_connect_rows.py`: 20 tests, 235-340 s;
- `Tests/UI/test_mcp_permissions_handoff.py`: 6 tests, 172-233 s.

Shards 2 and 11 also ran 19-20 minutes on #3225.

The census has grown from 430 to 664 entries since TASK-35834 re-dealt it on 2026-10-20. The workflow comment's rule for a shard nearing the cap is: replay the shard logs and re-order the census, not raise the minutes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The files that make the at-cap shards heavy are dealt over several shards, re-ordered by measured times from recent PR runs' shard logs
- [ ] #2 Every test the lane ran before the change still runs on it
- [ ] #3 Over at least three PR runs after the change, every UI Fast Lane job keeps at least 3 minutes under its 20-minute cap
<!-- AC:END -->
