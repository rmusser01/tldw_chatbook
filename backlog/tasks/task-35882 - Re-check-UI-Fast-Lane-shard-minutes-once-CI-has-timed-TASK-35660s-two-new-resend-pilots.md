---
id: TASK-35882
title: >-
  Re-check UI Fast Lane shard minutes once CI has timed TASK-35660's two new
  resend pilots
status: To Do
assignee: []
created_date: '2026-10-21 00:35'
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
- [ ] #1 scripts/ui_lane_shard_replay.py, run over the shard logs of at least two CI runs that include the two new pilots, is recorded in the notes with each shard's minutes
- [ ] #2 Every UI Fast Lane shard stays under its job limit with the margin the census comments ask for, or the census tail is re-ordered with --write and the replay re-run until it does
<!-- AC:END -->
