---
id: TASK-35834
title: UI Fast Lane job 9 runs within a minute of its 20-minute cap on dev
status: To Do
created_date: 2026-10-20 13:59
labels:
- follow-up
- ci
- ui-lane
priority: medium
references:
- scripts/ui_pr_gate_census.txt
- scripts/ui_lane_shard_replay.py
- .github/workflows/derived-artifacts.yml
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UI Fast Lane job 9 (`--shard 8 10`) has the least headroom of the ten jobs on dev's census, which is the census #3169 merged with.
- **The numbers.** It took 19.5 min wall (1,116 s of pytest) on run 39420530080 at #3169's head 03dd91f9ef, and 16.8 min (953 s) on run 39382891881 at 509f389f94. The job's cap, `timeout-minutes: 20`, includes setup.
- **One timeout already.** On run 39365877181 (ee15f29f4a, a slow runner), job 9 hit the cap and was cancelled while still in `test_settings_model_defaults_rows.py`.
- **The main cost** is that file, listed whole. It runs about 4.5-6 minutes on CI, all on this one job, and its tests top the job's "slowest 25". No test hangs.
- **Why it matters.** The workflow's own budget leaves about 3.3 minutes for a hung test's report, and job 9 has under a minute of it. Any PR that adds load to this shard's positions, or a slower runner, turns the required check red with no defect.

Found by the S1 rebase review (2026-10-20). The S1 PR keeps its own entries light on job 9, but it cannot fix dev's base load.


Additional evidence from [PR #3176](https://github.com/rmusser01/tldw_chatbook/pull/3176): [UI job 9 in run 39447290920](https://github.com/rmusser01/tldw_chatbook/actions/runs/39447290920/job/118712469761) exhausted its 20-minute limit on head 3c2b0e5601 while the final added paste journey was running. GitHub's check annotation explicitly names the 20m0s maximum; there was no assertion failure. That PR applies interim measured slot swaps and retains every entry and all limits. This does not distribute the heavy file by function, establish a three-minute margin, or supply three post-change runs; all criteria here remain open. Its qualification report records raw per-occurrence timings because repeated file names produced a negative cost in the stock replay helper.


Further qualification: [run 39455910928](https://github.com/rmusser01/tldw_chatbook/actions/runs/39455910928) at 916ea4e565 passed job 9 but jobs 4/6/8/10 exhausted the same 20-minute limit. Job 10 printed 176 passed before cancellation; no failing-test progress appears in these four logs. PR #3176 therefore profiles the full ten-job run and applies additional whole-file slot swaps with documented estimates for incomplete work. This is still interim qualification: it leaves the node-distribution, three-minute margin and three-run acceptance here open.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `test_settings_model_defaults_rows.py`, or whatever measures heaviest on job 9, is dealt over several shards, for example one census node per test function, re-ordered by measured times
- [ ] #2 Over at least three PR runs after the change, every UI Fast Lane job keeps at least 3 minutes under its 20-minute cap
- [ ] #3 `scripts/ui_lane_shard_replay.py`'s estimate is checked against measured shard times at that census, and corrected or documented if it still scores node-id entries by whole-file seconds (it was off by up to 6 minutes per shard on #3169's census, 2026-10-20)
<!-- AC:END -->
