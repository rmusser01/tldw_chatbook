---
id: TASK-37764
title: Commit a whole-census re-deal mode for the UI lane shard replay
status: To Do
assignee: []
created_date: 2026-10-30 03:57
labels:
- follow-up
- ci
- tests
priority: low
references:
- scripts/ui_lane_shard_replay.py
- Tests/CI/test_ui_lane_shard_replay.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`scripts/ui_lane_shard_replay.py` has no search that can re-deal a census this size. `--tail` accepts the whole entry count, but `flattest_tail` runs 300 restarts of an all-pairs swap loop, re-scoring the whole census after every swap, so in practice it cannot finish on the 1,149-entry census. And under the header rule, the first entry would take the header block with it. Every whole-census re-deal of the UI Fast Lane so far has used an uncommitted search, so none can be reproduced from the repo:
- TASK-36512's 13-shard deal (#3231);
- PR #3228's 17-shard deal;
- the 28-shard deal now on dev.

PR #3242 (TASK-36514, closed 2026-10-30 because dev had already moved to 28 shards) built a `--deal` mode:
- largest-first fill under round-robin counts, then flattening swaps, scored at each entry's worst observed time;
- deterministic, with comments moving with their entries;
- three tests in Tests/CI.

Its review found the tool sound but its determinism test unable to fail, because it re-runs the same input in the same process. The code is on branch `fu/h-15shards` (commit 10f59cf6e7) and in that PR's diff.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A committed replay mode re-deals the whole census for a given shard count from measured shard logs, and a second run on the same logs changes nothing
- [ ] #2 The deal's shard sets do not depend on the census's input order, and a test fails if they do
- [ ] #3 Comments stay with their entries, and the census header stays in place
- [ ] #4 On the logs it is given, the deal's heaviest shard, scored at each entry's worst observed time, is no heavier than the census as listed, and the tool prints both
<!-- AC:END -->
