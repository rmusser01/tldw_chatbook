---
id: TASK-34353
title: Shard the UI Fast Lane so the gated Tests/UI census fits its time cap
status: In Progress
assignee:
  - '@claude'
created_date: '2026-10-04 02:21'
updated_date: '2026-10-04 02:24'
labels:
  - ci
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The UI Fast Lane runs the whole PR-gate census serially under a 20-minute cap. It already measured 14-19 minutes, and as the census grows PRs time out with zero failing tests (#2992 three times; #2953, #2977 and #2993 also hit 20 minutes), so the required check goes red on wall-clock alone. Splitting the census into contiguous shards that run in parallel keeps every file gated and the order within each shard unchanged, and restores headroom under the cap.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The UI lane runs the census as parallel contiguous shards; every census file runs in exactly one shard, in census order
- [x] #2 A red or timed-out shard still turns the required 'Derived artifacts reproduce from their sources' check red
- [ ] #3 Each shard finishes well inside the 20-minute cap on a PR carrying the Phase 6 census additions
- [x] #4 Tests/CI pins the shard shape (contiguous slicing, fail-fast off, minimal deps, no xdist) instead of forbidding a strategy
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Split ui-fast-lane into a 2-shard matrix (fail-fast off); each shard slices census lines [index*n/total, (index+1)*n/total) from strategy.job-index/job-total, so shards are contiguous, in census order, and cover every file once.\n2. Keep the 20-min cap, minimal deps, no xdist; derived-artifacts' needs.ui-fast-lane.result already aggregates the matrix.\n3. Replace the Tests/CI 'no strategy' pins with shard-shape pins; negative control against dev's workflow.\n4. Measure shard times on this PR's CI; #2992 (Phase 6 census +5 files) rebases on it.
<!-- SECTION:PLAN:END -->
