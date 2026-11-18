---
id: TASK-39144
title: "Six UI Fast Lane shards already run 15.0-16.9 min of their 20 min cap, so rebalance or split the hot shards before the next census entry tips one over"
status: In Progress
assignee: ['@claude']
created_date: 2026-11-07 02:28
updated_date: '2026-11-17 21:12'
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

**Round-13 re-deal (2026-11-12, at the top of the round-13 stack, branch fu/r13-closeout).** AC#1 stays open until 12 PR runs after this deal.
- **Weights.** The cached runs the header named were gone with the old scratch directory, so the weights were re-read, read-only, from the 28 latest PR runs with all 36 UI shards finished: 42887475221 (2026-11-10 19:39Z) to 43248578539 (2026-11-12 15:30Z), 24 before #3500 and 4 after. Run 43085804799 is left out (its own change), and so are the 40 failed shard logs. Each entry is charged at its slowest run.
- **Charges for the 117 round-13 tail entries, none of which any CI log times.**
  - The cred stream's 71: its own charges (TASK-39154/39170, local serial at load 40-55).
  - 8 from their task notes, at the slowest serial time recorded there (load 44-90): TASK-39130's five nodes (42.8, 40.3, 22.5, 42.0 and 44.5 s), TASK-39153's `test_console_inspector_rail_minors.py` (94.0 s) and TASK-39171's two theme-picker nodes (38.5 and 46.4 s).
  - The other 38 (TASK-39169 13, TASK-40132 17, TASK-38948 7, TASK-39170 1) had no serial record. They were measured in one serial batch through run.sh with the UI lane's flags (`-p no:xdist --timeout=180 --durations=0`): 51 passed in 191 s at load 7-10.
  - **Calibration.** At that load, 14 entries already timed by the 28 runs, rerun serially in the same session, took 1.6-3.3 times as long in CI's slowest run as here (2.1 weighted by time; 3.3 and 15 on two sub-second entries). So the 38 are charged at their local seconds times 2.61, the top of the 3-10 s band 32 of them sit in. That is 186 s local, charged 485 s. At raw local seconds, the replay would have under-priced them; the lesson in `lessons-testing-evidence.md` records it.
- **Replay** (`scripts/ui_lane_shard_replay.py --shards 36`, through run.sh): as listed (the stacked census), max shard 17.7 pytest min (shard 17), with shard 5 at 16.6, which matches its 18.4-min CI job.
- **`--rebalance 13.3 --write`:** 24 swaps (48 entries moved). Every shard now replays at 13.2-13.3 pytest min, about 476 over 36, so about 14.9-15.0 min a job with setup.
- **Sensitivity checks.**
  - Charging the 38 at 2.11 instead also reached 13.3, in 21 swaps.
  - Charging them at 3.29 replays this deal at a max of 13.4, and 17 more swaps would bring it back to 13.3.
- **The margin is now gone.** The floor is 13.24 a shard, so the next tail entries cannot be absorbed by a re-deal. That needs the owner's shard-count decision (37 shards would replay at about 12.9).
- **Checks.** The entry set is unchanged (2028; `MINIMUM_FILES` stays 2028). The census header now describes this deal and how to charge on a quiet host.

**AC#1 after the round-13 deal (round-14 closeout, 2026-11-15; read-only `gh` on `derived-artifacts.yml`). Not met.** Round 13 merged at 2026-11-12 20:37Z (dev 4b90eb860b). Since then dev has only appended census entries at the tail (1455ceeb96 +1, 531acb8cdc +6), so every earlier entry keeps its shard. Read: every PR run created from 2026-11-12 20:30Z to the listing at 2026-11-15 00:58Z (134 PR runs with UI jobs), taking the 88 in which all 36 UI shards finished (success or failure). Since #3529 (merged 2026-11-14 17:22Z) PR jobs run on self-hosted runners (`tldw-ci-test`); that PR's own runs did too. The two runner kinds are kept apart:
- **GitHub-hosted (`ubuntu-latest`), 79 runs** (43306757484, 2026-11-12 21:05Z, to 43614165778, 2026-11-14 16:10Z). 25 of 36 shards passed 15 min at least once. The worst: shard 15 at 19.13 (run 43315451147), shard 3 at 18.53 (43461560641), shards 16 at 17.78, 5 and 19 at 17.53, 9 at 17.32 and 7 at 17.20. Shard 16 is the systematic one: median 14.72, over 15 in 37 of 79 runs. Shard 3 (19 of 79) and shard 5 (13 of 79) follow.
- **Self-hosted, 9 runs** (43519057842, 2026-11-13 22:24Z, to 43621261326, 2026-11-14 17:28Z; six are #3529's own PR runs, three came after it merged). Every shard's slowest job is 9.87-14.97 min: shard 3 14.97 (43620841946), shard 16 14.90 (43621134977), shard 5 14.57. So all 36 are 5 min under the cap, but with 0.03-0.43 min to spare on shards 3, 5 and 16, and over 9 runs, not the 12 this criterion asks for. Six more self-hosted runs had fewer than 36 shards finished.
- **Reading.** On the hardware PRs now use, the deal holds the margin so far, barely. On GitHub-hosted runners (which PRs return to if the routing is reverted) it does not: shard 16 sits on the 15-min line. This criterion stays open until 12 self-hosted runs with every shard finished are recorded (9 so far). The census also grew by this round's tail entries since the deal, so the next read should name the deal it measures.

Per shard (minutes, slowest finished job; runs as above):

| Shard | GitHub-hosted slowest (run) | median | runs over 15 | Self-hosted slowest (run) |
|---|---|---|---|---|
| 1 | 15.65 (43352647859) | 12.90 | 3 of 79 | 14.03 (43620841946) |
| 2 | 15.58 (43487098385) | 12.22 | 1 of 79 | 12.28 (43539534151) |
| 3 | 18.53 (43461560641) | 13.25 | 19 of 79 | 14.97 (43620841946) |
| 4 | 15.55 (43324072512) | 12.92 | 1 of 79 | 12.98 (43621134977) |
| 5 | 17.53 (43398660691) | 14.15 | 13 of 79 | 14.57 (43620841946) |
| 6 | 14.33 (43322446796) | 11.78 | 0 of 79 | 12.45 (43539534151) |
| 7 | 17.20 (43368849944) | 12.95 | 1 of 79 | 12.92 (43539534151) |
| 8 | 15.42 (43461093022) | 12.93 | 3 of 79 | 13.72 (43621134977) |
| 9 | 17.32 (43347732794) | 13.10 | 3 of 79 | 13.35 (43620841946) |
| 10 | 14.92 (43377188324) | 12.45 | 0 of 79 | 12.72 (43621261326) |
| 11 | 14.97 (43376469788) | 12.40 | 0 of 79 | 12.50 (43548543910) |
| 12 | 15.60 (43379840549) | 12.20 | 1 of 79 | 12.38 (43612634465) |
| 13 | 15.77 (43461231048) | 12.50 | 2 of 79 | 13.25 (43557511010) |
| 14 | 16.18 (43547251067) | 13.37 | 6 of 79 | 13.08 (43620841946) |
| 15 | 19.13 (43315451147) | 12.80 | 8 of 79 | 13.32 (43557511010) |
| 16 | 17.78 (43491262486) | 14.72 | 37 of 79 | 14.90 (43621134977) |
| 17 | 15.58 (43369049379) | 12.53 | 2 of 79 | 13.02 (43621134977) |
| 18 | 15.27 (43551904218) | 11.93 | 1 of 79 | 12.07 (43621261326) |
| 19 | 17.53 (43468362487) | 12.75 | 5 of 79 | 12.42 (43621134977) |
| 20 | 14.82 (43461231048) | 12.77 | 0 of 79 | 13.13 (43620841946) |
| 21 | 16.12 (43461231048) | 12.63 | 2 of 79 | 13.05 (43621261326) |
| 22 | 13.55 (43330912705) | 11.77 | 0 of 79 | 11.90 (43620841946) |
| 23 | 14.12 (43595218971) | 12.15 | 0 of 79 | 12.82 (43620841946) |
| 24 | 15.42 (43324066851) | 12.72 | 1 of 79 | 12.65 (43621261326) |
| 25 | 15.98 (43398660691) | 12.73 | 1 of 79 | 12.87 (43620841946) |
| 26 | 15.98 (43461198245) | 12.55 | 3 of 79 | 13.40 (43548543910) |
| 27 | 13.07 (43307798365) | 10.60 | 0 of 79 | 10.92 (43620841946) |
| 28 | 12.25 (43598709405) | 9.87 | 0 of 79 | 9.87 (43620841946) |
| 29 | 15.22 (43347732794) | 12.10 | 2 of 79 | 12.15 (43519057842) |
| 30 | 15.60 (43382204664) | 13.03 | 1 of 79 | 13.50 (43620841946) |
| 31 | 14.67 (43461198245) | 11.70 | 0 of 79 | 12.32 (43621134977) |
| 32 | 16.33 (43395893969) | 12.35 | 1 of 79 | 12.00 (43621134977) |
| 33 | 15.42 (43598709405) | 12.22 | 2 of 79 | 12.22 (43621134977) |
| 34 | 15.60 (43346597851) | 13.15 | 5 of 79 | 13.20 (43621134977) |
| 35 | 13.42 (43567900224) | 11.45 | 0 of 79 | 11.68 (43621134977) |
| 36 | 14.50 (43491660299) | 12.45 | 0 of 79 | 12.82 (43620841946) |

**Round-14 closeout, Part D (2026-11-15, the round-14 stack on dev ecad404bd4): 38 shards, 19 at once.**
- **Owner decision, relayed by the coordinator:** "38 shards, 19 at once".
- **Change.** `ui-fast-lane`'s matrix goes from 36 to 38 shards and `max-parallel` from 18 to 19. Every shard-count statement follows: the workflow comment, the census header (`Current UI lane`, `round-robin over`), the checker's `Current UI lane` line, the replay's usage lines (`--shards 38`, `--rebalance 13.0`), the two `Tests/CI` matrix pins (`test_ci_queue_pressure_contract.py::_assert_ui_capacity`, `test_derived_artifacts_workflow.py`), and ADR-103 (dated amendment; 50 eligible test jobs, 31 concurrent test runners, 32 with perf-guard, from dev's 48 / 30 / 31).
- **Weights.** Read-only `gh`: the 25 latest PR runs with all 36 UI shards finished, 2026-11-14 03:20Z (43551098837) to 19:32Z (43632652505): 17 GitHub-hosted, 8 self-hosted (43557511010, 43565526841, 43612634465, 43620841946, 43621134977, 43621261326, 43628541939, 43632652505; checked per run with read-only `gh`, review r14cl-rev1). The per-kind offsets below were computed per job from each job's runner, so the split error did not reach them. Failed shard logs (33) were left out, so 867 shard logs. Each entry is charged at its slowest run. The 2034 entries dev carries are all timed by those logs.
- **Charges.** All 111 round-14 tail entries, each at its local serial time recorded in its own task's notes; no entry lacked a record. 2,174.8 s in all (36.2 min). Per task: TASK-40311 254 / 43 / 87 s (loaded 38-47); TASK-40712 33 / 18; TASK-40308 30 / 34; TASK-40320 23.3 and 16.5 (whole files); TASK-40319's 20 nodes 24.8-55.5 s and TASK-40318's 13 cases 34.0-62.2 s (load 55-67); TASK-40322's 10 nodes 0.3-12.5 s and TASK-40323's 48 nodes 0.3-9.7 s (setup + call + teardown, load 19-26); TASK-40324 7.8; TASK-40293 31.9, 15, 15; TASK-40327 24.4; TASK-40328 14.4, 11.9, 28.2; TASK-40300 24; TASK-38948 9.43 / 6.59 (this closeout, load 34). Two of the 111 (the 100 ms nodes) were also timed in some PR logs (6.3 / 6.7 s); the charge replaces that.
- **Replay** (`scripts/ui_lane_shard_replay.py`, through run.sh): the census carries 492.8 worst-case pytest min, a 13.69 floor at 36 and 12.97 at 38. As listed it replays at 10.4-17.9 at 36 and 9.4-19.3 at 38. `--rebalance 12.9` cannot be reached (113 swaps, it ends at the floor). `--shards 38 --rebalance 13.0 --write` took 46 swaps (90 entries moved, 6 of them round-14's) and every shard now replays at **12.94-12.99 pytest min**. Entry set and comments unchanged (2145, floor 2145).
- **Projected job time.** Model check on the census before round 14 (dev's 2034, 36 shards, same logs): the measured slowest job per shard ran a median 1.82 min above its replay on GitHub-hosted runners (0.57 to 4.25) and 0.58 min above on self-hosted ones (-1.03 to 3.08). Per-job setup (job minutes less pytest's own seconds) is a median 0.99 min GitHub-hosted and 1.46 self-hosted. So the 38-shard deal projects **about 14.8 min a job at the median on GitHub-hosted runners and about 13.5 on self-hosted ones**; the high ends (up to about 17.2 and 16.1) come from runs whose own change slowed a shard, such as `codex/shared-console-mobile-pr`.
- **AC#1 stays open.** It needs 12 PR runs after this change, which only exist after the merge. Before it, there are 11 complete self-hosted PR runs since round 13 (the 9 counted in Part A plus 43628541939 and 43632652505; 8 of the 11 are in the weight set above), and the newest, 43632652505 (2026-11-14 19:32Z, `codex/shared-console-mobile-pr`), finished shards 16, 19 and 3 at 15.73 (failed), 15.68 and 15.62 min: not 5 min under the cap. The other new one, 43628541939, peaked at 14.08 (shard 3).
- **No margin left.** At 38 the floor (12.97) is the target, so the next tail entries need another shard-count decision; the census header says so.
- **Checks** (run.sh, 2026-11-15). Census checker: OK, 2145 entries, floor 2145. `Tests/CI` at `-n 4`: 1640 passed, 1 skipped (load 44-48); its shard-count statements pin derives 38 shards, 50 eligible test jobs, 31 runners and 32 with perf-guard from the workflow. Size ratchets (module, screen, Library modules), serial: 282 passed. Preflight passed.

**Round-15 closeout (2026-11-17, the round-15 stack on dev 5e3c8b2bd4, top 212f015a91): 42 shards, 21 at once.**
- **Owner decision (2026-11-16):** "42 shards, 21 at once (Recommended)".
- **Change.** `ui-fast-lane`'s matrix goes from 38 to 42 shards, and `max-parallel` from 19 to 21. The same statements as Part D follow:
  - the workflow comment;
  - the census header;
  - the checker's `Current UI lane` line;
  - the replay's usage lines;
  - the two `Tests/CI` matrix pins;
  - ADR-103, with a dated amendment.

  The library-shell push lane's comment now says it borrows a third of the UI lane's runners (7 of 21), not half.
- **ADR-103 counts.** They are derived from the workflow the way `Tests/CI/test_ui_lane_shard_count_statements.py::_counts` does.
  - The aggregate's `needs` holds 12 non-preflight jobs: the UI lane plus 11 others. Console P0 adds 2 extra profiles.
  - That gives 42 + 11 + 2 = **55 eligible test jobs**, and min(42, 21) + 11 + 2 = **34 concurrent test runners**, **35** with perf-guard's one job.
  - Dev 5e3c8b2bd4 itself was at 51 / 32 / 33 before this change; TASK-41217's `plugins-recovery-ui` (2af759be6c) is a dev job.
- **Weights.** Read-only `gh` over the 33 latest PR runs whose 38 UI shards all finished (success or failure) and whose jobs GitHub served: 2026-11-15 18:54Z (43749365761) to 2026-11-17 09:38Z (44057405523).
  - Runners: 27 GitHub-hosted, 4 self-hosted (43749365761, 44008003165, 44017381806, 44057405523), and 2 with one self-hosted job each (43918311026, 43932909359).
  - 27 failed shard logs were left out, so 1227 shard logs. Each entry is charged at its slowest run.
  - 43898439593 returned HTTP 502 on every attempt and is not in the set. 44017381806 also returned 502 at first; it was added once GitHub served it (37 success, 1 failure).
- **Charges.** The 97 entries present at 212f015a91 and absent at dev 5e3c8b2bd4: 96 tail entries plus TASK-41130's in-place rename. 2024.6 s in all (33.7 min). Dev's own two entries (TASK-41218's node, TASK-41248's file) are priced by the CI logs: 1.6 s and 2.7 s.
  - From each task's own notes:
    - TASK-41130's renamed node: 51.9.
    - TASK-41125: 24.3.
    - TASK-41139: 42.3, and its visit file 86.2 (43.3 + 42.9).
    - TASK-41102: 54.9, then 57.1, 42.5, 57.8 and 71.2. The conversation node takes its later run (54.9 s at load 72-92) in place of 71.2.
    - TASK-41126: 46.3 and 14.3. The 46.3 s is the post-fix-round serial time in its notes, replacing 65.6; the round's ledger records it too.
    - TASK-41304: 48.5 and 50.8. Its notes record call time only, with no setup or teardown, so these stay call time.
    - TASK-41136: 5.4, 0.1, 0.1, 0.3, then nine nodes at 20.0-63.2.
    - TASK-41138: its per-entry table. Later measurements replace three entries: late voice 12.2 (10.0 + 2.2), real Console [False] 98.6 and [True] 102.1 (the top of each range). The cold-home node takes 36.6, its fix-round-2 re-measurement, in place of 35.0-99.5.
  - TASK-41127 / TASK-41303's 48 entries.
    - Their notes give per-entry times from two serial lane-flag processes: the 36 earlier entries at load 7-35, and the 12 later ones from the 733.6 s all-48 run at load 71-55.
    - The round's ledger records the later rev2 re-measurement (load 47-63) only as totals: 312.7 s for the 36, 242.0 s for the 12. Each group's per-entry times are scaled to those totals: 274.3 s to 312.7 s, and 214.8 s to 242.0 s.
    - The 733.6 s run is not used per entry for the 36 earlier entries, because it recorded no per-entry times for them, only the run's total. Its 12 per-entry times serve only as that group's proportions, and rev2 supersedes the run itself.
  - **Literal history** (`test_library_prompt_mixins.py::test_literal_history_restore_keeps_kind_definition_and_guards`) is charged **140 s**, not its scaled 101.6 s (review r15co-rev1, M1).
    - TASK-41127's notes say to charge per-process times on a loaded host, giving 140 s for mixins at load 63-105. The reviewer's lane-flag run at load about 85 timed it at 202.7 s.
    - The per-process rule is applied only here. This entry is long enough that a process's start-up is a small share of its 140 s. For the other entries, the notes' 6-63 s per-process times are mostly that start-up, which a CI shard pays once, not per entry.
- **Replay** (`scripts/ui_lane_shard_replay.py`, through run.sh).
  - The census carries 519.6 worst-case pytest min: a 13.67 floor at 38, 12.99 at 40 and 12.37 at 42.
  - As listed it replays at 8.91-17.00 at 38 and 8.08-19.74 at 42.
  - `--shards 42 --rebalance 13.0 --write` took 19 swaps with the first charges, then 1 more after review r15co-rev1's re-charge. A single run from the base would not take that path; the head deal is kept. Together they moved 39 entries, 5 of them round 15's.
  - Every shard now replays at **11.66-12.80 pytest min**, against a floor of 12.37.
  - The entry set is unchanged (2243, floor 2243).
- **Projected job time.** Model check on dev 5e3c8b2bd4's census (2147 entries at 38, replayed from the same logs at 8.15-14.63).
  - The measured slowest job per shard ran a median 1.73 min above its replay on GitHub-hosted runners (0.12 to 2.46, over 1100 jobs).
  - It ran 0.47 min below on self-hosted ones (-1.58 to 1.43, over 154 jobs).
  - So the 42-shard deal projects at most about 14.5 min a job at the median on GitHub-hosted runners (15.3 at the high end) and about 12.3 on self-hosted ones.
- **Calibration of local charges** (reviews r15co-rev1 and r15co-rev2, recorded only). Each round-14 entry is priced as the replay prices it (`entry_seconds`), on dev's census, where all 111 still exist.
  - Round 14's 111 charges, 2174.8 s local and loaded, took 867.3 s of CI worst case over these runs: CI is **0.40x** of the local charges.
  - Per entry, CI over local: p90 2.33x and max 6.6x (a 0.3 s entry).
  - For the 63 entries of 5 s or more: p90 1.27x and max 1.67x. The maximum is `test_console_controller_wiring.py::test_composer_accessor_is_late_bound`, 5.5 s local against 9.2 s CI.
  - Reproduced with the reviewer's `v2-calib.py`. Over 32 runs it gives 866.3 s, the same ratios.
  - Fix round 1 recorded 908.8 s, 0.42x, p90 1.35x and max 2.41x. That figure summed node durations directly, which double-counts the two whole-file round-14 entries (`test_workspace_archive_async_completion.py`, `test_workspace_archive_review_regressions.py`): their file time plus their nodes' phase rows, about 41.5 s more. Those two entries alone produced the 2.41x maximum.
  - The 2.6x quiet-host rule and the margin statement are unchanged.
- **The 38-shard deal in CI** (context for AC#1, not its evidence). Over the same runs:
  - GitHub-hosted jobs' slowest per shard was 10.03-15.83 min; 11 shards passed 15 min at least once (shard 34 in 4 runs, 29 in 3).
  - Self-hosted jobs peaked at 14.62 (44017381806).
- **AC#1 stays open.** It needs 12 PR runs after this change, which only exist after the merge.
- **Margin.** At 42 the floor is 12.37, about 0.63 pytest min a shard, so about 26 min of new entries fit before the next shard-count decision. The census header says so.
- **Checks** (run.sh, 2026-11-17, after reviews r15co-rev1 and r15co-rev2). Census checker: OK, 2243 entries, floor 2243 (same entry set, no duplicates). `Tests/CI` at `-n 4`: 1661 passed, 1 skipped (load 67-79); its shard-count statements pin derives 42 shards, 55 eligible test jobs, 34 runners and 35 with perf-guard from the workflow. Size ratchets (module, screen, Library modules), serial: 283 passed. Preflight passed. Ruff gate passed.
