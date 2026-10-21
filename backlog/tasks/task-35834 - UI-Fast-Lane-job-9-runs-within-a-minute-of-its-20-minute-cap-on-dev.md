---
id: TASK-35834
title: UI Fast Lane job 9 runs within a minute of its 20-minute cap on dev
status: In Progress
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
assignee:
- '@codex'
updated_date: 2026-10-21 00:03
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md (existing). Reason: tune the existing serial round-robin census and its diagnostic estimator; preserve dependencies, triggers, required aggregation and timeout contracts. 1. Preserve PR3182 head8e6b1ee timing evidence; measure actual per-occurrence costs and setup overhead, and identify heavy files. 2. Add failing replay controls for repeated file occurrences and node-level timing; correct estimator without changing gated assertions. 3. Replace measured heavy whole-file entries with complete function-level coverage, rebalance using measured costs, and validate exact collected-node preservation. Keep20-minute job and180-second test limits; add matrix capacity only if measured budgets require it, updating existing count statements. 4. Run targeted CI controls and static checks; request independent Superpowers review and fix findings. Add work to PR3182 and rebase onto current dev with force-with-lease. 5. Obtain at least three successful post-change PR run attempts with all UI jobs at most1020seconds; compare replay against actual census and publish qualification. Address PR feedback, mark Done only after every criterion is evidenced.
<!-- SECTION:PLAN:END -->


## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented complete16-function coverage of model-defaults rows (same 23 collected parameter nodes), a measured13-shard deal with 430 census entries, full duration logging, and replay attribution for repeated files and unequal node costs. The92-entry Phase7 block is rebalanced without dropping any prior selector or assertion; job20m/test180s limits, dependencies and required aggregation remain unchanged. Existing ADR-103 is amended with dated capacity evidence; no new ADR required.

Four replay regressions and the split-coverage guard failed correctly before their fixes. Replay14PASS; six affected CI files390PASS with one stale matrix-size pin, then the corrected pin1PASS. Exact collection and census430/floor430 pass. Old-log node estimates retain documented partial-duration limitations; corrected exact-census baseline estimates differ from observed shard progress by at most33seconds. Rebased onto dev2d071d6d71 without conflicts. All391 affected CI controls pass; scoped Ruff/format/whitespace and census pass. Superpowers review found no blocker; both Minor findings were corrected (spaced parameter IDs and canonical census hash), with26 replay/count checks passing. Three hosted qualification attempts remain pending; acceptance stays unchecked until evidenced. See Docs/superpowers/qa/2026-10-20-ui-lane-headroom.md and companion evidence JSON.
<!-- SECTION:NOTES:END -->
