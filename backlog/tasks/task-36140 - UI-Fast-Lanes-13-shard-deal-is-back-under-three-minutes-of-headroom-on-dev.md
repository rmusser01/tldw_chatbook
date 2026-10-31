---
id: TASK-36140
title: UI Fast Lane's 13-shard deal is back under three minutes of headroom on dev
status: To Do
assignee: []
created_date: '2026-10-22 17:59'
labels:
  - ci
  - ui-lane
  - follow-up
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-35834 (PR #3182, merged 2026-10-21 15:07Z) moved the UI Fast Lane to 13 shards and qualified it with at least 3 minutes under the 20-minute job cap on three runs. Dev's census has since grown from 430 entries (at #3182's merge, fd4df0a534) to 584 (at 7700c63a68), and the margin is gone again.
- **Measured job wall minutes**, four green PR runs on the 13-shard deal, 2026-10-22: run 39730705223 max 18.1 (job 3); 39734506492 max 17.4 (job 4); 39744284533 max 17.5 (job 2); 39799596144 max 16.7 (job 3). Three of four leave under 3 minutes.
- **Replay.** scripts/ui_lane_shard_replay.py --shards 13 over those four runs at 7700c63a68's census: 13.8 15.8 16.8 16.2 15.5 15.8 13.8 15.8 14.1 14.7 13.7 14.8 12.8, max 16.8 pytest minutes. With TASK-35834's 110 s of setup that is about 1,118 s, above its 1,000 s projection target.
- **Why it matters.** On the ten-shard deal this overrun evicted PR #3188 (TASK-35920) and PR #3190 (run 39610893391, attempts 2 and 3: job 2 'exceeded the maximum execution time of 20m0s' at 97% with no failing test, and with none of the PR's changed entries in that job). A PR that adds load to a heavy job, or a slow runner, turns the required check red with no defect.

Found while diagnosing PR #3190's eviction (TASK-35660), 2026-10-22.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Replayed over at least three recent 13-shard (or later) runs, every UI Fast Lane job projects at or under TASK-35834's 1,000 s target, or the census is re-dealt or the matrix widened (per ADR-103) until it does
- [ ] #2 Over at least three PR runs after the change, every UI Fast Lane job keeps at least 3 minutes under its 20-minute cap
<!-- AC:END -->
