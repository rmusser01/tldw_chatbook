---
id: TASK-39144
title: "Six UI Fast Lane shards already run 15.0-16.9 min of their 20 min cap, so rebalance or split the hot shards before the next census entry tips one over"
status: In Progress
assignee: ['@claude']
created_date: 2026-11-07 02:28
updated_date: '2026-11-12 19:40'
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
- Owner decision (2026-11-06), as relayed by the coordinator: "Accept, record exception … file a follow-up to rebalance or split the hot shards themselves". Under it, round 11 put one small census entry on each of four of these hot shards.
- Every further tail entry is dealt by position, not by time, so the next one can land on a hot shard and push it to the cap. A timed-out shard fails the required check.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every UI Fast Lane shard's slowest finished job, over at least 12 PR runs after the change, is at least 5 min under its cap; the per-shard figures and runs are recorded before and after
- [x] #2 The approach is named in the notes (re-ordering entries within the census by measured weight, splitting heavy entries, adding shards, or a combination), with the measured entry weights it used
- [x] #3 Every census entry still runs exactly once per PR, and the census checker and `Tests/CI` pass
- [x] #4 The census header records how new tail entries keep the 5 min margin, so the next addition does not re-derive it
<!-- AC:END -->

## Implementation Plan

1. Collect per-shard UI job minutes and every shard log of at least 12 recent PR runs (read-only `gh`), and build per-entry weights from the logs' progress lines and `--durations=0` rows (`scripts/ui_lane_shard_replay.py`).
2. Check the replay against the measured slowest job per shard, and compute the floor (all work / 32 shards + setup).
3. Extend `scripts/ui_lane_shard_replay.py` with a whole-census `--rebalance` that swaps the fewest entries between shards (the checker's `read_census`/`shard` still deal), and re-deal the census with this round's tail entries charged.
4. Add shards only if no re-deal brings every shard's projected slowest job 5 min under the cap.
5. Record the before/after projections and add the census header note; AC#1's 12 post-change PR runs need the merge.

## Implementation Notes

**Owner decision (2026-11-06):** "Reorder by weight first: re-deal the census by measured per-entry CI time so heavy entries spread out; no extra runners. Add shards only if reordering can't reach the 5-min margin."

**Approach (AC#2): re-deal by measured weight, plus two shards, because no order of 32 reaches the margin.**
- Weights. Every UI shard log of 22 PR runs, read-only (`gh api .../jobs/<id>/logs`): 42321107301, 42320698779, 42319781834, 42310788537, 42309351181, 42306963606, 42294368927, 42287112198, 42270955674, 42268842238, 42267936447, 42263533424, 42253456601, 42252461355, 42233793917, 42233786797, 42233778418, 42233324796, 42233321199, 42218508176, 42210918177, 42202410121 (2026-11-06 10:24Z to 2026-11-07 00:55Z; 42209855017 left out, 16 shards red on its own change). Each entry costs its slowest run, from the logs' progress-line stamps and `--durations=0` rows (`scripts/ui_lane_shard_replay.py`'s `entry_seconds`, unchanged). The UI lane passes `--durations=0 --durations-min=0`, which overrides pyproject's `--durations=25`.
- 106 entries no log timed (round 11's 61 tail entries, merged without a CI run, and this round's 45) are charged at their serial seconds here with the UI step's command, at load 50-155. Twelve already-timed entries run in the same local run took 0.51-0.99 times as long in CI as here (median 0.73), so 1.0 times local is the upper end. Charged total: 34.7 min.
- Model check. Replayed pytest minutes plus 1.7 min (the median of job minutes minus logged file seconds over all 704 shard jobs; p90 1.87) against each shard's slowest measured job: shard 5 18.51 vs 18.63, shard 3 17.84 vs 17.65; the measured minus the model runs from -2.16 (shard 30) to +0.85 min (shard 29), median about -0.6: the worst-case-per-entry sum leans high.
- Floor. The census carries 448 worst-case pytest minutes: 13.9 min a shard at 32 shards, about 15.6 min a job, so no order of 32 keeps every job 5 min under the cap. The whole-census swap search reached 14.0 at 32, 13.6 at 33, and 13.2 at 34 (44 swaps); 35 would reach 12.7.
- Tool. `scripts/ui_lane_shard_replay.py` gained `--rebalance MINUTES` (a greedy search that swaps entries between shards until no shard's replay passes MINUTES; `--write` moves each entry with the comment lines right above it, leaving blank lines and block-opening entries in place) and `--charge ENTRY=SECONDS`. Dealing stays the checker's own `read_census`/`shard`, which `shard_loads` now calls too. Two `Tests/CI/test_ui_lane_shard_replay.py` tests pin the swap search and the rewrite; red under a temporary edit that skipped the target check and the comment walk.
- Applied: `--shards 34 --rebalance 13.2 --write` (44 swaps; see the review note below for the final deal), with the matrix at 34 shards and every shard-count statement updated (workflow, census, checker, replay usage, ADR-103: 44 eligible jobs). `max-parallel` stays 17, so peak runners do not change.

**Per shard, minutes (AC#1's before; after is projected):**

| Shard | Slowest job, 22 runs before | Before: replay + 1.7 (32 shards, base census) | After: replay + 1.7 (34 shards) |
|---|---|---|---|
| 1 | 14.93 | 15.92 | 14.89 |
| 2 | 14.63 | 15.07 | 14.87 |
| 3 | 17.65 | 17.84 | 14.88 |
| 4 | 14.05 | 14.67 | 14.90 |
| 5 | 18.63 | 18.51 | 14.89 |
| 6 | 13.02 | 13.51 | 14.88 |
| 7 | 14.47 | 15.11 | 14.88 |
| 8 | 15.28 | 15.81 | 14.88 |
| 9 | 13.98 | 14.51 | 14.90 |
| 10 | 13.92 | 13.89 | 14.88 |
| 11 | 14.60 | 14.91 | 14.88 |
| 12 | 14.40 | 14.14 | 14.90 |
| 13 | 16.07 | 16.14 | 14.88 |
| 14 | 13.93 | 14.35 | 14.87 |
| 15 | 15.02 | 15.34 | 14.87 |
| 16 | 14.87 | 15.40 | 14.89 |
| 17 | 13.33 | 14.72 | 14.87 |
| 18 | 14.62 | 16.29 | 14.88 |
| 19 | 15.00 | 16.20 | 14.90 |
| 20 | 14.30 | 14.48 | 14.88 |
| 21 | 13.95 | 14.75 | 14.88 |
| 22 | 13.13 | 13.94 | 14.90 |
| 23 | 14.20 | 14.39 | 14.88 |
| 24 | 15.87 | 16.47 | 14.87 |
| 25 | 12.62 | 13.46 | 14.87 |
| 26 | 14.70 | 15.92 | 14.90 |
| 27 | 13.78 | 14.54 | 14.90 |
| 28 | 14.50 | 15.76 | 14.87 |
| 29 | 15.60 | 14.75 | 14.89 |
| 30 | 13.45 | 15.61 | 14.90 |
| 31 | 13.80 | 14.60 | 14.88 |
| 32 | 12.38 | 13.95 | 14.88 |
| 33 | - | - | 14.88 |
| 34 | - | - | 14.88 |

The after column is 14.87-14.90 on every shard with the untimed entries at 1.0 times local; at the median ratio (0.73) it would be about 14.6. That margin is thin: the model put one shard 0.85 min under its measured slowest job. If post-merge runs show a shard over 15, 35 shards replays at 12.7 + 1.7.

**Spot check.** Shard 30 of the new deal (54 entries, 8 of them moved by the swaps), run serially in lane order with the UI step's command: 87 passed in 920 s at load 33-53. A re-deal changes which entries run before each entry; the first CI run of all 34 shards is the full check.

**AC#1 (open).** It asks for 12 PR runs after the change, which only exist after the merge.

**AC#3.** `scripts/check_ui_pr_gate_census.py`: OK, 1854 entries, floor 1854, no duplicates: the swaps only reorder. `Tests/CI` at `-n 4`: 1228 passed, 1 skipped (2026-11-07, final tree). Preflight passed; the boot budget ratchets (`bootbudget.sh`): 32 passed.

**AC#4.** The census header now says how a new tail entry keeps the margin: charge it at its local serial time, replay, and run `--rebalance 13.3 --write` if a shard passes 13.3 pytest min.

**Files.** `scripts/ui_lane_shard_replay.py`, `Tests/CI/test_ui_lane_shard_replay.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`, `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`, `Tests/CI/test_derived_artifacts_workflow.py`, `backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md`, `backlog/docs/lessons-testing-evidence.md`.

**Review r12l-rev1 (2026-11-08).** Sixteen more entries joined the tail (the inspector focus carriers file by node and the Console full config preparation file whole, out of the recovered-suites job), and the census now carries 452 worst-case pytest min (122 untimed entries charged at their local serial seconds). The floor at 34 shards is 13.29 pytest min; `--rebalance 13.3 --write` took 35 more swaps (79 in all) and projects 14.99-15.00 min a job on every shard, about 14.7 at the median CI-to-local ratio. That meets the 5-min margin with almost no headroom; 35 shards would project about 14.6. Floor 1870. The census header and `--charge`'s help now agree (local serial time), and the header calls the search greedy. Spot checks in lane order: shard 20, 107 passed in 955 s; shard 34, 109 passed in 895 s (load 36-58). `Tests/CI` at `-n 4`: 1228 passed, 1 skipped; preflight passed; census checker OK at 1870.

**Owner decision (2026-11-07):** "36 shards, 18 at once". The reason given to the owner: once the other round-12 branches' tail entries join (about 27), the census carries about 465 worst-case pytest min, so 34 shards project about 15.3 min a job and no re-deal reaches the 15-min margin; 36 at `max-parallel: 18` keeps two waves (no PR wall-clock change), projects about 14.6 min, and costs one more peak runner. Applied on this branch: matrix 36 shards, `max-parallel: 18`, and every pinned statement (workflow comment, census header, checker, replay usage, `Tests/CI` pins, ADR-103: 46 eligible jobs, 28 test runners in parallel and 29 with perf-guard on this branch, which has no other branch's added job). `--rebalance 13.3 --write` at 36 took 22 swaps from the re-dealt order (as listed at 36 the heaviest shard replayed 21.4 min, because a new count re-deals every entry). On this branch's census (1870 entries, 452 worst-case pytest min) that projects 13.28-14.98 min a job, about 12.9-14.7 at the median CI-to-local ratio. The target only stops the search at 13.3 pytest min, so the spread stays uneven: from this order, `--rebalance 12.9` would take 7 more swaps to 14.6 min a job everywhere, and `--rebalance 12.6` 22 more to 14.3. The coordinator re-deals again on the stacked tip.

**Round-12 restack (2026-11-08).** The coordinator restacked the six round-12 branches onto dev b590a15afd; the census kept this branch's 36-shard deal whole and took 37 tail entries from dev and the other branches (1907 entries, floor 1907). Of those, 5 (the TestPersonaHumanIdentityRemoval nodes) have CI timings in the 22 runs' logs (#3419's PR runs); the other 32 are charged at their local serial time, measured through run.sh in one batch with the UI step's command (84 passed in 301 s at load 20-27; per entry 0.6-48.6 s, 283 s in all). The census then carries 457 worst-case pytest min: a 12.69-min floor over 36 shards, so the asked `--rebalance 12.6` cannot be reached (it ends at the floor after 95 swaps). `--rebalance 12.7 --write` took 40 swaps and puts every shard at 12.68-12.70 pytest min: **14.38-14.40 min a job** with setup, about 13.7-14.3 at the median CI-to-local ratio. As listed after the restack the heaviest shard replayed 13.8 (about 15.5 min a job). The 32 entries were measured at a lower load than the earlier charges (20-27, against 50-155), so their charge is closer to CI's own time. ADR-103's runner arithmetic, recomputed from the stacked workflow: 11 lanes, 36 shards, `max-parallel: 18`: 46 eligible test jobs, 28 test runners in parallel, 29 with perf-guard, as ADR-103 says.

**Restack onto dev 3a6498ae14 (2026-11-09).** The census kept the 12.7 deal and took dev's 7 TASK-34569 approval entries at the tail (1913 entries). The earlier weights were lost with the scratch directory, so they were re-read, read-only, from the 22 most recent PR runs whose 32 UI shards all finished (42538427422, 2026-11-09 00:07Z, to 42715332317, 22:49Z; 42539733304 left out, 28 of its shards red on its own change). Those logs time dev's 7 entries; the 92 entries no log timed (all round-12 entries) are charged at their local serial time, measured in one run.sh batch with the UI step's command (154 passed, 1 failed, 851 s at load 10-29; 0.3-28.3 s per entry). The census carries 430 worst-case pytest min (floor 11.9 a shard over 36). As listed the heaviest shard replayed 13.7; `--rebalance 12.7 --write` took 12 swaps and projects **13.1-14.4 min a job** (11.4-12.7 pytest min plus 1.7), about 12.9-14.3 at the median CI-to-local ratio. The one failure is not this branch's: `Tests/UI/test_cancelled_dialog_waits.py::test_cancelling_the_waiting_worker_leaves_the_dialog_closable[media-analysis-delete]` (TASK-39146, another round-12 branch) imports `tldw_chatbook.Widgets.Media.media_viewer_panel`, which TASK-38605 (dc456a948e, on dev) deleted; 3 of 3 reruns fail with ModuleNotFoundError.

**AC#1, CI after the change (round-13 closeout, 2026-11-12; read-only `gh` on derived-artifacts.yml PR runs). Not met.** Round 12 merged with the 36-shard deal at about 15:40Z on 2026-11-10 (dev ae9f9e8d21). #3500 (merged 15:06Z on 2026-11-12) removed two census entries, which shifts every later entry's shard, so runs are split around it.
- **Before #3500:** 33 PR runs (2026-11-10 19:39Z to 2026-11-12 14:58Z), 24 of them with all 36 shards finished (42887475221 to 43242842744). Run 43085804799 is left out: its own change put shards 34 and 35 at 19.2 and 17.9 min. Slowest finished job per shard: 12.62-16.32 min. Four shards passed the 15-min mark:
  - shard 5: 16.32 (run 43192096387), median 15.08 over 34 jobs;
  - shard 21: 15.70 (run 43142676395);
  - shard 3: 15.68 (run 42919238652);
  - shard 35: 15.15 (run 43138509441).
- **After #3500:** 7 PR runs so far, 4 with all 36 finished (43244450534, 43245032382, 43248233114, 43248578539). Slowest per shard: 11.05-18.38 min. Over the mark: shard 5, 18.38 (run 43248233114, median 15.73 over 7 jobs), and shard 22, 15.05.
- **Before the change** (the table above): 22 runs at 32 shards, slowest per shard 12.38-18.63 min.
- **Reading.** The deal holds the margin on 31-32 of 36 shards, but shard 5 is over it in most runs (its median is above 15 min). Shard 5's 18.38-min job (131298357463) ran 351 passed in 1021.5 s of pytest. No single entry dominates: the slowest are `test_library_reader_console_delete_refresh.py`'s two delete cases, 22.9 s each.
- **Round 13 makes it heavier.** The census goes from 1913 to 2028 entries (117 added, 2 removed). Under the append rule (`ui_pr_gate_census.txt:17-20`), 46 of the additions were never charged and replayed:
  - TASK-39169: 13
  - TASK-40132: 17
  - TASK-38948: 7
  - TASK-39130 and TASK-39153: 6
  - TASK-39171: 2
  - TASK-39170: 1

  Only the cred stream's 71 were rebalanced, on its own branch at base 1913, and the restack then shifted them by 11 slots. The header (`:11-16`) still describes the 11-09 deal. A charge, replay and `--rebalance 13.3 --write` of the stacked census is due before the round-13 PR, and shard 5 is the one to relieve. This criterion then needs 12 PR runs after that deal.
