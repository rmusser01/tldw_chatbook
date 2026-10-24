---
id: TASK-36514
title: Run the UI Fast Lane as 16 shards so a hung test's report fits under the cap
status: In Progress
assignee:
- '@claude'
created_date: 2026-10-23 23:38
updated_date: 2026-10-24 17:52
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

**Owner ruling 2026-10-24:** the lane runs as 16 shards, not 15: in the 12-run worst-case model a perfect 15-shard deal scored 17.38 minutes a job against AC#3's 17.0 limit, and 16 scored 16.43.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The UI Fast Lane runs as 16 shards, and the census is re-dealt for 16 from measured shard logs
- [x] #2 Every test the lane ran before the change still runs on it, exactly once
- [ ] #3 Over at least three PR runs after the change, every UI Fast Lane job keeps at least 3 minutes under its 20-minute cap
- [x] #4 The comment above ui-fast-lane records the shard count and the measurement it rests on
- [x] #5 The re-deal can be reproduced with a tool committed to the repo
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read the shard logs of every usable PR run (the nine pre-deal runs, the three on TASK-36512's deal, and any newer post-deal run). Compute the worst-case floor for each shard count.
2. Give `scripts/ui_lane_shard_replay.py` a `--deal` mode that re-deals the whole census for N shards:
   - it minimises the heaviest shard with each entry at its slowest run (largest-first fill, then improving swaps);
   - it is deterministic;
   - each entry keeps its comment lines and its census order inside its shard.
   Tests go in `Tests/CI/test_ui_lane_shard_replay.py`.
3. Rewrite the census comments so each one stays true wherever its entry lands: no slot, shard, tail or block-placement claims. Add a fixed header that states the order rules.
4. Try listing `test_settings_advanced_disclosures.py` by function. Keep the split only if the replay improves, and if kept, add it to `SPLIT_CENSUS_FILES`.
5. Make the edits that go with the new count:
   - set the matrix to 16;
   - re-deal with the tool;
   - set `MINIMUM_FILES` to the entry count;
   - update the comment above `ui-fast-lane`, the shard-count statement contract, ADR-103's counts and the workflow contract test.
6. Verify, all through run.sh:
   - collection equality of the old and new censuses (`--collect-only -q --keep-duplicates`);
   - the split guard;
   - `Tests/CI` with `-n 4`, the census checker and preflight.
7. Leave AC#3 to three post-change PR runs.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Measured.** UI Fast Lane shard logs from 13 PR runs:
- nine pre-deal runs (40024354185, 40007935895, 40007779176, 40003892794, 40000099679, 40018273828, 39995171485, 39994223628, 39993112363);
- three on TASK-36512's 13-shard deal (40043746975, 40100579406, 40107677135);
- one newer post-deal run (40109377531, a 17-shard branch).

Run 40108848893 returned HTTP 502 on every fetch. Scored with each entry at its slowest run, the census holds 228.5 pytest minutes. The floors are 15.2 at fifteen shards and 14.3 at sixteen. Setup before pytest took 62-117 s over 173 jobs. The worst job therefore comes to 17.4 minutes at fifteen and 16.5 at sixteen, against the 17.0 target (owner ruling: sixteen).

**Tool (AC#5).** `scripts/ui_lane_shard_replay.py --shards 16 --deal --write <run dirs>` re-deals every entry. Shard s keeps its round-robin count. Entries fill the lightest shard largest first, then pairwise swaps flatten the loads, each entry scored at its worst run. Ties go by entry name, never census order, so the same logs give the same census, and a second deal changes nothing. Inside a shard, entries keep census order. Each entry moves with the comment lines above it through the existing `_reordered`. `--tail` works as before. The deal takes about 0.2 s.

**Census (AC#1).** The comments that described a slot, a shard, the tail or a block were rewritten or dropped, so each comment stays true wherever its entry lands. Every TASK-36512 "keep this block where it is" header is gone. A new header states the order rules, and a blank line ends it so it stays first. `test_settings_advanced_disclosures.py` (300-457 s) is now listed by its 20 functions and added to `SPLIT_CENSUS_FILES`.
- Split, it changes nothing in the worst case: both deals reach the floor.
- The 13 runs' own heaviest shard drops from 13.9 to 13.6.
- Also splitting `test_settings_connect_key_rows.py` or `test_settings_console_fallback_rows.py` kept the floor and moved per-run figures only within the noise between deals, so they stay whole.

The matrix is 16, and `MINIMUM_FILES` follows the entry count. Replayed worst case: 14.3 on every shard, against 17.2-18.9 on dev's 13-shard census. The runs' own heaviest shard is 12.1-13.6, against 15.5-18.4.

**AC#2 evidence.** Through run.sh, `pytest --collect-only -q --keep-duplicates` over the old census's entries and the new census's entries collected 3,257 node ids each, with no duplicates. The sorted lists are byte-identical (sha256 a7188692...). The split guard was red on the old census ("test_settings_advanced_disclosures.py is listed whole again") and is green on the new one.

**AC#4.** The comment above `ui-fast-lane` records sixteen shards and the measurement. The shard-count contract now reads the TASK-36514 statements: the workflow comment, the replay's three usage lines and ADR-103's counts (22 runners, 23 counting perf-guard). Its number words also accept digits past twenty. The census and checker statements are now history and are no longer pinned. `test_derived_artifacts_workflow.py` pins `range(1, 17)`.

**Red/green evidence.**
- `dealt` temporarily returning the census unchanged failed both deal tests: `assert (8.03 < 6.1)`, and the written census differed.
- The header test failed on dev's census, which had no header.
- With the matrix temporarily back at 13, the shard-count and workflow contracts gave 11 failed.
- All are green now.

**Checks.**
- `Tests/CI -n 4`: 1034 passed, 1 skipped.
- Census checker OK.
- Preflight passed.
- Ruff: no new findings, and the changed test files are formatted.

**Order risk.** The deal changes which files run before which in every shard, and only CI can show a test that relied on a predecessor.

**AC#3 is open.** It needs three post-change PR runs.

Files: `.github/workflows/derived-artifacts.yml` (matrix and comment), `scripts/ui_lane_shard_replay.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`, `Tests/CI/test_ui_lane_shard_replay.py`, `Tests/CI/test_ui_pr_gate_census_checker.py`, `Tests/CI/test_ui_lane_shard_count_statements.py`, `Tests/CI/test_derived_artifacts_workflow.py`, `backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md`, `backlog/docs/lessons-testing-evidence.md`.
<!-- SECTION:NOTES:END -->
