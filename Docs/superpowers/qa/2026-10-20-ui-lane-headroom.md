# UI Fast Lane headroom — TASK-35834

Scope: extend [PR #3182](https://github.com/rmusser01/tldw_chatbook/pull/3182), preserving every previously gated test and its assertions. Governing [ADR-103](../../../backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md) and completed TASK-34353/33664.4 apply; no new architecture decision is needed. [Machine-readable evidence](2026-10-20-ui-lane-headroom-evidence.json) binds baseline logs, costs, layouts and the census.

## Diagnosis and change

Baseline PR head `8e6b1ee4280b62b7330f02f2bbd35b963bcb8e5c`, run39509516931, passed all ten UI jobs but the worst used 1183/1200seconds. The model-defaults whole-file entry carried about 367seconds in that run. The stock replay reused the final timestamp for repeated file names: a three-occurrence control charged90s and negative 30s instead of 40s and 20s. It also split every file equally among node selectors regardless of measured node durations.

The replay now retains each occurrence independently, reads setup/call/teardown duration rows, includes parameter cases under function selectors and keeps unequal measured node costs. Older top-25 logs leave unknown work: only the remaining file total is split among unmeasured selectors, or as shared overhead when all selectors have some measurement. A partially reported parameterized function can still be underestimated in those historical logs; these estimates are placement evidence, not exact node profiles. New UI runs print all duration phases with no minimum cutoff. File/node worst cases are conservative and can add up to more than any single run.

All 16 model-defaults test functions replace the whole-file entry; every parameter case stays gated. Census415->430, floor430. Only the existing92-entry Phase7 block is rebalanced (first entry fixed); the remaining15 function selectors append. Other prior selectors and the relative order outside that block are retained. A regression guard requires all test functions and a multi-shard deal. Actual pytest collection confirms the same 23 unique parameterized nodes for whole-file versus all function selectors.

Thirteen shards are the smallest evaluated layout that meets the 1020s target with 110s setup allowance and 15% variance on two green runs' worst costs: ten 1254s, twelve 1063s, thirteen 1000s, fourteen 1027s. The14-way deal is worse because fixed entries change buckets. These are estimates, not hosted qualification. Matrix count statements now distinguish current capacity from historical dated counts. Unchanged:20-minute job cap,180-second test timeout, minimal dependencies, serial execution per job, fail-fast:false, required aggregation and event triggers.

## Verification and closure gate

Four replay regressions failed by assertion on the old implementation; the separate coverage guard failed because one whole-file entry replaced16 expected function selectors. All 14 initial replay tests then passed; the rebased six-file CI contract run passes all 391 tests in 80.07s. The six affected CI contract files produced390passes and one expected stale ten-shard pin failure; the pin now expects thirteen and its isolated rerun passes. Corrected historical replay compared against each exact census (408entries at5a5c59e991,415entries at8e6b1ee428) differs from progress-line shard measurements by at most33seconds, instead of the old multi-minute misattribution. The source JSON retains all ten errors per run. The first pytest invocation stopped at the Windows shared-writable OS Temp directory before collection; subsequent runs use a task-local TEMP/TMP and retain native private-path checks. Existing pytest-asyncio, Pydantic and optional-audio warnings remain visible.

Targeted CI contracts, static checks, exact collection, independent Superpowers review, current-dev rebase and three post-change hosted PR attempts are required before closure. Run attempts use the real pull_request workflow, not workflow_dispatch; no dummy source changes or raised timeouts. The completed qualification receipts appear below. No application/UI behavior changes or full-suite sweep are part of this CI task.


## Rebased source review

The branch rebased without conflicts onto dev `2d071d6d7145184ba41408ec437306cfb188ea57`. The six affected CI contract files pass all 391 tests in 80.07s. Scoped Ruff and formatting pass; whitespace and census checks pass. Superpowers isolated review of task commit `3c4d13dfb1` found no Critical/Important issue and independently reproduced coverage and layout costs. Both Minor findings were corrected: a new RED/GREEN control preserves spaces in parameter IDs, and the evidence hash now uses the canonical LF/UTF8 Git blob after rebase. All 26 replay/count checks pass after those fixes. Reviewer scope exclusions have explicit controller dispositions in the evidence JSON. This source review preceded hosted qualification; the completed receipts appear below. No merge is authorized.

Scoped Superpowers re-review of `194b522440` confirms both Minor findings resolved and no actionable residual finding. That source approval preceded the successful three-attempt timing and replay qualification below.

## Completed hosted qualification

Three fresh attempts of the real pull-request workflow ran at source head 6719cfa5f71cc132a6f6db34e0c4f56e13962601, after the requested rebase onto dev 2d071d6d7145184ba41408ec437306cfb188ea57. All 39 UI job IDs are distinct. Every job succeeded within 1020 seconds, and all three required artifact aggregates succeeded. At third-attempt capture only the nonrequired Merge queue tick was pending; its status is retained separately.

| PR attempt | UI jobs | Slowest job | Minimum headroom | Maximum replay error | Required aggregate |
| --- | --- | --- | --- | --- | --- |
| [1](https://github.com/rmusser01/tldw_chatbook/actions/runs/39520530263/attempts/1) | 13/13 | 16m00s | 4m00s | 0.475s | Passed |
| [2](https://github.com/rmusser01/tldw_chatbook/actions/runs/39520530263/attempts/2) | 13/13 | 15m06s | 4m54s | 0.450s | Passed |
| [3](https://github.com/rmusser01/tldw_chatbook/actions/runs/39520530263/attempts/3) | 13/13 | 15m56s | 4m04s | 0.448s | Passed |

All 39 checkout logs identify merge 1d73ce51d93119ee468d8ea0e38faa787a52eed7; its census bytes match the 430-entry source census and canonical LF/UTF8 SHA256 4c78b9e9c20a869b937080ad60e7d121ba4c1e229e9f97a5bffaf4a512ff6cd0. The companion JSON retains each job ID/link, wall/progress/pytest/replay timings, duration-row counts, and the SHA256 of each available workflow log archive captured after the required jobs completed.

Job wall time includes setup, tests and cleanup, and excludes waiting for a runner. Replay error compares the estimator with actual pytest progress at that exact census; it does not treat setup as test work. Every shard was checked, with maximum absolute errors below 0.5 seconds in each attempt. The 20-minute job cap, 180-second test limit, dependencies, coverage and required aggregation remain unchanged.

The independent Superpowers evidence audit found no actionable issue and verified all 39 timing, checkout and census bindings. Its attempt-3 aggregate observation was pending at inspection; the controller retained the final successful required-aggregate receipt and available workflow archive before closure. Live PR feedback and the fresh documentation-closure head are checked separately after publication. [Perf Guard](https://github.com/rmusser01/tldw_chatbook/actions/runs/39520529951) also passed on the qualified source head. No full-suite sweep was run.


## Final integration rebase

Dev advanced by 19 commits while qualification ran. The final integration rebase onto dev 906497d9b4db2388df515f6d8f332273f051466f replayed all 20 branch patches without conflicts; every patch was unchanged before this documentation amendment. Workflow, census, replay/checker and four affected CI control files retain identical Git blobs and the same canonical census hash.

The pure rebased tree exactly matches the already-tested PR merge fe9fce1a802e07506f17a707aa1c200f6c3caf44. An actual completed UI checkout log independently binds that merge to the pre-integration closure-head CI: required aggregate and Perf Guard passed, and all 13 additional UI jobs passed with at least 3m49s headroom. Combined with the three qualification attempts, 52 distinct UI jobs meet the three-minute target. The original three-attempt/source-bound evidence above remains the qualification record.

This final amendment only records integration evidence. Fresh required CI on the published rebased head is reported in the PR description after completion; these source records do not claim that new head has already passed.
