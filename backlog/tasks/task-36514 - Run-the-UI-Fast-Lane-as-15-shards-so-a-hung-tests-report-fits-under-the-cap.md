---
id: TASK-36514
title: Run the UI Fast Lane as 15 shards so a hung test's report fits under the cap
status: Won't Do
assignee: []
updated_date: 2026-10-30 03:58
created_date: 2026-10-23 23:38
labels:
- follow-up
- ci
- tests
priority: high
references:
- .github/workflows/derived-artifacts.yml
- scripts/ui_pr_gate_census.txt
- scripts/ui_lane_shard_replay.py
dependencies:
- TASK-36512
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-36512 (2026-10-23).

**The lane no longer fits in thirteen shards with headroom.** TASK-36512 replayed nine PR runs' UI Fast Lane shard logs. Scored with each entry at its slowest run, the census holds 223.6 pytest minutes. No order can put less than 17.2 minutes on the heaviest of 13 shards. Each job also spends 1.0-2.0 minutes (median 1.4) on setup before pytest starts, so the worst case is about 19.2 minutes against the 20-minute cap.

**That leaves no room for a hung test's report.** A hung private-profile test costs about 3m10s-3m18s before the wrapper reports it (the workflow's own comment above `ui-fast-lane`). The 3 minutes of headroom TASK-35834 kept is therefore gone.

**More shards bring the floor down:** 16.0 minutes at 14 shards, 14.9 at 15, 14.0 at 16.
- In the worst-case model, 15 shards leave about 0.1 minutes of margin against a 17-minute target even when perfectly balanced: the floor of 14.9, plus up to 2.0 of setup. 16 shards (floor 14.0) leave about 1 minute. Per-run replays are lower. If replays show 15 cannot meet AC#3, the shard count goes back to the owner before implementation.
- The shipped `scripts/ui_lane_shard_replay.py --write` re-orders only the tail. TASK-36512's re-deal used an uncommitted scratch search, so a full re-deal has no repo tool yet.

**Owner ruling 2026-10-23:** ship TASK-36512's 13-shard re-deal now, then move the lane to 15 shards with a full re-deal. The matrix lives in `.github/workflows/derived-artifacts.yml`, which draft PR #3036 also changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The UI Fast Lane runs as 15 shards, and the census is re-dealt for 15 from measured shard logs
- [ ] #2 Every test the lane ran before the change still runs on it, exactly once
- [ ] #3 Over at least three PR runs after the change, every UI Fast Lane job keeps at least 3 minutes under its 20-minute cap
- [ ] #4 The comment above ui-fast-lane records the shard count and the measurement it rests on
- [ ] #5 The re-deal can be reproduced with a tool committed to the repo
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Won't Do, overtaken (2026-10-30).
- **The work.** PR #3242 built the change: first 16 shards (owner ruling 2026-10-24, after 15 failed the 13-run worst-case gate at 17.38 min against 17.0), then a rebase to keep 17. That rebase was needed because #3228 had moved dev's lane to 17 shards meanwhile; the owner ruled to keep 17 and re-deal with the tool.
- **Why it was dropped.** Before the rebase finished, dev moved the UI Fast Lane to 28 shards (1,149 entries) through other PRs, which gives far more headroom than this task asked for. #3242 was closed.
- **What survives.**
  - The test leak its review found, where the response-rules fixture leaves a llama.cpp endpoint in the shared bootstrap config, was already fixed on dev by TASK-36478 (#3254, 2026-10-26). That change restores the shared bootstrap config after every bootstrap-profile test, and a negative control with that restore disabled reproduced the egress error.
  - The committed whole-census re-deal mode it built is filed as TASK-37764.
<!-- SECTION:NOTES:END -->
