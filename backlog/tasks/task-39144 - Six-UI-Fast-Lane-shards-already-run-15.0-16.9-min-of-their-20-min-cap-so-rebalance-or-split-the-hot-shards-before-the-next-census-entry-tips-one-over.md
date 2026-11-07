---
id: TASK-39144
title: "Six UI Fast Lane shards already run 15.0-16.9 min of their 20 min cap, so rebalance or split the hot shards before the next census entry tips one over"
status: To Do
created_date: 2026-11-07 02:28
labels:
- follow-up
- ci
- tests
priority: medium
references:
- scripts/ui_pr_gate_census.txt
- scripts/check_ui_pr_gate_census.py
- scripts/ui_lane_shard_replay.py
- .github/workflows/derived-artifacts.yml
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 11 of the 2026-10-23 burn-down; checked on dev 2684e5453f and against CI, 2026-11-07. Line numbers are dev's.

**The lane.**
- The UI Fast Lane job ("Run the gated Tests/UI slice") runs 32 shards, each with `timeout-minutes: 20` (`derived-artifacts.yml:857`, `:860`).
- It deals census entries round-robin: entry i goes to shard i mod 32 (`scripts/check_ui_pr_gate_census.py --shard`).

**The measurement.** The slowest finished job per shard over the last 12 PR runs, read from the GitHub API on 2026-11-07 01:30Z (runs 42247976071, 2026-11-06 15:58Z, to 42310788537, 23:16Z):

| Shard | Slowest run |
|---|---|
| 3 | 16.87 min |
| 5 | 16.70 min |
| 13 | 16.07 min |
| 8 | 15.28 min |
| 15 | 15.02 min |
| 19 | 15.00 min |

- Across all 32 shards the slowest-run figures run from 11.8 min to 16.9 min (median 14.05). Shards 16, 1, 26 and 28 sit at 14.5-14.9 min.
- One spot check matches the data: run 42310788537's shard 3 took 23:37:59 to 23:49:50 (11.85 min), the figure recorded for that run.

**Why now.**
- Owner decision (2026-11-06), as relayed by the coordinator: "accept, record exception … file a follow-up to rebalance or split the hot shards themselves". Under it, round 11 put one small census entry on each of four of these hot shards.
- Every further tail entry is dealt by position, not by time, so the next one can land on a hot shard and push it to the cap. A timed-out shard fails the required check.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every UI Fast Lane shard's slowest finished job, over at least 12 PR runs after the change, is at least 5 min under its cap; the per-shard figures and runs are recorded before and after
- [ ] #2 The approach is named in the notes (re-ordering entries within the census by measured weight, splitting heavy entries, adding shards, or a combination), with the measured entry weights it used
- [ ] #3 Every census entry still runs exactly once per PR, and the census checker and `Tests/CI` pass
- [ ] #4 The census header records how new tail entries keep the 5 min margin, so the next addition does not re-derive it
<!-- AC:END -->
