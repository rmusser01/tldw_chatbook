---
id: TASK-36512
title: UI Fast Lane shards 3 and 11 time out at the 20-minute cap on PR runs
status: In Progress
assignee:
- '@claude'
created_date: 2026-10-23 22:46
updated_date: 2026-10-23 23:10
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
- [x] #1 The files that make the at-cap shards heavy are dealt over several shards, re-ordered by measured times from recent PR runs' shard logs
- [x] #2 Every test the lane ran before the change still runs on it
- [ ] #3 Over at least three PR runs after the change, every UI Fast Lane job keeps at least 3 minutes under its 20-minute cap
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Download the UI Fast Lane shard logs of recent PR runs (one directory per run, outside the worktree); measure each job's setup (job start to pytest's `collected` line) and check the replay against each run's own census.
2. Score every census entry at its slowest run with `scripts/ui_lane_shard_replay.py`'s functions; find the entries that make shards 2, 3 and 11 heavy.
3. Replace those whole-file entries with their collected function selectors (or parametrize cases for a one-function file): one selector keeps the file's slot, the rest are appended at the census tail. Compare a tail-only deal with one that also re-deals the Phase 7 block (whose header already allows `--write` re-ordering), and keep the better replay.
4. Prove with `pytest --collect-only -q`, shard list by shard list, that the old and new censuses collect the same node set. Extend the existing split-file guard in Tests/CI to the new split files, red before the census change.
5. Raise `MINIMUM_FILES` to the entry count; run the census checker, the Tests/CI census/replay controls and preflight through run.sh.
6. Leave AC#3 to post-change PR runs.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Measured.** Shard logs of nine PR runs from 2026-10-23 (40024354185, 40007935895, 40007779176, 40003892794, 40000099679, 40018273828, 39995171485, 39994223628, 39993112363). Eight of them ran this branch's census unchanged. Replayed over each run's own census, the replay matched the measured shard pytest time to within 1 s on seven runs. Setup, from job start to pytest's `collected` line, took 62-117 s (median 85 s). With each entry scored at its slowest run, the heaviest shard carried 19.6 pytest minutes. The heavy whole-file entries on the at-cap shards were `test_settings_connect_rows.py` (235-367 s), `test_mcp_permissions_handoff.py` (172-243 s), `test_settings_connect_tab_budget.py` (188-323 s) and `test_settings_default_model_picker.py` (229-394 s).

**Change.** Those four files are now listed by function, and the MCP handoff by its six parametrize cases, since it has one test. Each file keeps one selector in its old slot, and the other selectors are appended at the census tail. The tail selectors and the 92-entry Phase 7 block were then dealt together, by a swap search. The search minimised the heaviest shard with every entry at its slowest run; ties went to the lowest heaviest shard in the nine runs' own replays. Every entry outside the Phase 7 block and the tail keeps its slot.

The worst-case replay is now 17.2-17.3 pytest minutes on every shard, against 15.8-19.6 before. In the nine runs' own replays, the heaviest shard is 15.5-16.7 minutes, against 16.4-19.2.

**Deviation: the Phase 7 block is re-dealt.** The brief preferred a tail-only deal. Replayed, the tail-only deal only reached 18.3: round-robin gives every shard an equal share of the tail, and each split selector costs at least one app boot (~15 s), so shards already at the average gain 1-2 minutes. The block's header already allows `--write` re-ordering, and TASK-35834 re-dealt the same block. The tail was not ordered by the stock `--tail --write`, which searches only the tail. Splitting four more heavy files did not help (18.2).

**AC#2 evidence.** `pytest --collect-only -q` of each of the 13 old and 13 new shard lists, through run.sh, collects 3,101 node ids each, with no id twice. The sorted sets are byte-identical (sha256 bbc5c49d...). The census checker resolves every new case id.

**Guard.** The model-defaults split guard in Tests/CI is now `test_a_split_census_file_keeps_every_test_gated_once`, parametrized over the five split files. A function must be listed, or else all of its literal parametrize cases. Red/green evidence:
- Red on the old census: 4 failed, 1 passed, on the "listed whole" assertion.
- Two negative controls on the new census both fired: a dropped connect-rows function ("is not in the census") and a dropped MCP case ("misses a case", 5 == 6).
- Green after the change.

**AC#3 is open, and not expected to pass at thirteen shards.** No ordering beats the sum of every entry's slowest time divided by the shard count: 223.6 / 13 = 17.2 pytest minutes. With up to 1.9 minutes of setup, the worst case projects to 18.6-19.2 minutes a job, not 17. Even averages fall short. The runs averaged 13.8-15.1 pytest minutes a shard, so a perfectly balanced shard still takes about 15.3-16.6 minutes a job with setup. The lightest shard of the slowest run then needs every job within about 0.4 minutes of that average to stay at 17. The floor is 16.0 minutes at 14 shards and 14.9 at 15, so the 3-minute target needs about 15 shards with a full re-deal, or less work per shard. The change removes the at-cap shards. It does not provide that headroom.

**Checks.** Census checker OK. `Tests/CI` with `-n 4`: 1029 passed, 1 skipped. Preflight passed. Ruff check and format pass on the changed test file; the checker's two ruff findings predate this change.

Files: `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py` (floor), `Tests/CI/test_ui_pr_gate_census_checker.py`, `backlog/docs/lessons-testing-evidence.md`.
<!-- SECTION:NOTES:END -->
