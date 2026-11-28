---
id: TASK-38716
title: "PR Fast Lane and the Console P0 gate run at their time caps"
status: In Progress
assignee:
- '@claude'
created_date: 2026-11-04 00:59
updated_date: 2026-11-28 10:00
labels:
- follow-up
- ci
- needs-owner-decision
priority: high
references:
- .github/workflows/derived-artifacts.yml
- Tests/CI/test_ci_queue_pressure_contract.py
- backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 6 of the 2026-10-23 follow-up burn-down. Checked on dev c2cdb6b82e, 2026-11-03. The figures below are job wall times (`started_at` to `completed_at`), read with `gh api` from the 200 most recent `derived-artifacts.yml` pull-request runs (jobs started 2026-11-01 09:13Z to 2026-11-04 00:43Z).

**PR Fast Lane** (`pr-fast-lane`, `timeout-minutes: 35`, derived-artifacts.yml:255):
- Jobs started on 2026-11-03: 44 successful runs, median 31.5 min, p95 34.5, max 34.7. Including the 25 failed runs: median 31.8, p95 34.3. Two runs were cancelled at 35.0 and 35.5 min, which is the cap.
- The whole window: 52 successful runs, median 31.4, p95 34.5, max 34.7.

**Console P0 regression gate** (`console-p0-gate`, `timeout-minutes: 20`, :940):
- Jobs started on 2026-11-03: 56 successful runs, median 18.2 min, p95 19.6, max 20.0. Nine more were cancelled at 20.2-20.9 min, consistent with the cap cancelling them.

**Stale timing comments.**
- The `pr-fast-lane` comment (:246-254) still records TASK-34916's replay: "median 16.1 / max 20.8 min, 14 min under the cap". The inline note on its timeout cites a 27m46s worst case.
- The `console-p0-gate` comment (:924-936) estimates about 9 min and records TASK-34986's "median 11.6 / max 13.9".

Both jobs are back where TASK-34916 and TASK-34986 started. Those tasks fixed it by moving steps into a new job, and the comment's own rule is "add a job (a shard), not minutes". Every new job costs one more concurrent runner per PR (ADR-103). Runner capacity is the owner's call, so the remedy needs the owner's decision first: a shard, a split, or a cap change.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The owner's decision on the remedy (shard, split or cap change) is recorded in the task notes
- [ ] #2 Over at least 20 pull-request runs after the change, the PR Fast Lane's p95 job time is at least 5 minutes under its cap and the Console P0 gate's is at least 4 minutes under its cap; the window and the figures are recorded in the task notes
- [x] #3 No test leaves the required check to achieve it
- [x] #4 Both jobs' timing comments in the workflow state the measured median and maximum and the window they came from, and the contract tests that pin the lanes reflect the new layout
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Check that nobody is on it: open pull requests and remote refs that edit the job, dev's history of its cap, the board on every ref.
2. Read every PR Fast Lane job since 2026-11-26 from the API and its log, and per-file times. Sort each cap cancellation into slow or hung. Compare fast and slow logs over the files they share.
3. Cost each way out against the measured runner speeds on today's target list: a higher cap, a job of its own for the heaviest file, a move to a lane with room, xdist workers.
4. Put the options to the owner. A cap raise and an added runner have both been the owner's decision before (ADR-103).
5. Apply the choice with its `Tests/CI` pins and a dated ADR-103 note, run `Tests/CI` and preflight, then measure 20 full pull-request runs (AC #2).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-11-05 CI evidence, added during round-8 filing. No remedy is decided here; AC#1 still needs the owner's decision.

**PR Fast Lane, as read by the reviewer.** On 2026-11-05 the reviewer read 54 `PR Fast Lane` jobs that started between 2026-11-04 04:49Z and 2026-11-05 20:34Z.
- Successful runs had a median of 32.0 min and a maximum of 34.8 min, against the 35-minute cap.
- 8 jobs were cancelled at the cap.

**Filer's cross-check, first attempts only.** The filer read the first attempt of each `derived-artifacts.yml` pull-request run with `gh api`, giving 41 `PR Fast Lane` jobs in the same window:
- 13 successful: median 32.0 min, maximum 34.8 min.
- 10 failed.
- 17 cancelled: 7 at 35.3 min (the cap), 8 before reaching it, and 2 at 54.2 and 60.2 min.
- 1 skipped.

**Console P0 regression gate.** It was cancelled at its 20-minute cap on #3412: job 127602265822 in run 42127370254, branch fu/r8-endpoint, started 23:32:24Z, cancelled 23:52:42Z (20.3 min).

---

**2026-11-28, re-measured (PR Fast Lane).** The Console P0 half of this task was settled by TASK-34986, TASK-40724 and TASK-42533 (ADR-103, 2026-10-14, 2026-11-13 and 2026-11-24); this note is the PR Fast Lane's. Plan steps 1 to 4 are done, and step 5 was applied the same day (below). The first commit of the pull request that carries this note left the workflow unchanged, so its own CI was the light tier; the second applies the owner's choice, which makes it the full run. The recovered suites are TASK-42413's and have their own note.

**Owner decision (AC #1), 2026-11-28: "Apply my picks".** The picks were option 4 here, a job of its own for the two MCP workbench files (one more runner), and option 2 for the recovered suites (TASK-42413). Applied in the pull request that carries this note: `pr-fast-lane-mcp-workbench` ("PR Fast Lane (MCP workbench)", 20-minute cap, the lane's install line and flags), the two files out of `pr-fast-lane`, the aggregate's `needs` and a verdict step for the new job, the pins in `Tests/CI`, and ADR-103's counts and dated note. The PR Fast Lane keeps its 35-minute cap.

**Checked first.** No open pull request (41) changes this job's cap. Of 236 remote refs with the workflow file, 13 edit the job's block against their merge base with `origin/dev`: six change one line (`runs-on`, `if` or `python-version`; PRs 3342 and 3579 and four branches with no PR); four `fix/task-34000-*` branches add riders that dev already has through squash merges; `codex/windows-post-save` (PR 3534, last pushed 2026-11-17) adds 14 riders; `codex/ci-bounded-qualifications` is an old restructure; and `codex/shared-console-mobile-pr` (PR 3345, a draft pushed today) takes `Tests/UI/test_mcp_workbench.py` out of this job (its 279 nodes join the UI census), adds four riders (`Tests/TTS/test_audio_player.py`, `Tests/DB/test_base_db_connection_lifecycle.py`, two `Tests/MCP/test_stdio_*` files) and sets `PYTEST_ADDOPTS` on the step. Only `codex/ci-bounded-qualifications` touches the cap line. `git log origin/dev -S"owner decision 2026-10-09, raised 30 -> 35"` finds only 57bfcb29aa, which set it. This task is the only one on any ref about this job's cap: TASK-34916 (Done) split its admission-sensitive half out, and TASK-42413 and TASK-42719 are the recovered suites' and the Dreams riders job's.

**Measured.** GitHub API job records of the 467 `pull_request` runs of `derived-artifacts.yml` created from 2026-11-20 to 2026-11-28 05:50 UTC, each job execution counted once (a rerun of failed jobs repeats the others' records). A cancellation is "at the cap" only when the job's annotation says "The job has exceeded the maximum execution time of 35m0s": all 11 below were read. From 2026-11-26, 86 jobs reached a verdict or the cap: 75 finished (median 31.2, p90 33.9, max 34.7 minutes) and 11 were cancelled at 35 minutes. The finished jobs' median has climbed from 29.1 on 2026-11-20 to 31.5 on 2026-11-27, and the job's targets went from 167 on 2026-11-18 to 177 on 2026-11-27:

| UTC day | finished | median | p90 | max |
| --- | --- | --- | --- | --- |
| 2026-11-20 | 41 | 29.1 | 31.8 | 33.4 |
| 2026-11-22 | 49 | 29.1 | 31.8 | 33.6 |
| 2026-11-24 | 48 | 29.9 | 32.7 | 34.8 |
| 2026-11-26 | 35 | 31.2 | 33.9 | 34.7 |
| 2026-11-27 | 36 | 31.5 | 33.3 | 34.5 |

**The 11 cancellations are two different things.** Each cut job's log was read for its last progress line and the time from it to the cancel.

| Run (job) | Branch (PR) | Started UTC | Log shows | Job would have needed |
| --- | --- | --- | --- | --- |
| 45758071217 (139741082248) | fix/estop-holds-agent-tool-calls | 11-26 07:52 | slow: progressing, cut at 95% | about 36.4 min |
| 45894250185 (140165828969) | perf/task-33620.15.55-admission-entry-waits (3686) | 11-26 21:04 | slow: cut at 90% | about 38.5 |
| 45913835197 (140232152353) | ci/42626-tier-sample | 11-27 00:11 | slow: cut at 97% | about 35.2 |
| 45931006623 (140285096965) | ci/42626-tier-sample | 11-27 03:10 | slow: cut at 90% | about 38.5 |
| 46016195678 (140598371701) | codex/shared-console-mobile-pr (3345) | 11-27 15:56 | **hung**: silent for 11.3 min after 98% | no end |
| 46036486304 (140637995167) | fu3/41959-inspector-order | 11-27 17:30 | slow: cut at 92% | about 37.6 |
| 46052364475 (140695978665) | codex/shared-console-mobile-pr (3345) | 11-27 19:33 | **hung**: silent for 14.0 min after 98% | no end |
| 46060245493 (140732676251) | codex/task-40608-windows-admission (3575) | 11-27 21:22 | slow: cut at 97% | about 35.1 |
| 46075621675 (140770936166) | codex/shared-console-mobile-pr (3345) | 11-27 23:13 | **hung**: silent for 13.1 min after 98% | no end |
| 46093295331 (140827843457) | codex/shared-console-mobile-pr (3345) | 11-28 02:27 | **hung**: silent for 14.5 min after 98% | no end |
| 46099491652 (140849778807) | codex/shared-console-mobile-pr (3345) | 11-28 03:33 | **hung**: silent for 19.5 min after 98% | no end |

- **Five are one branch's hang, not slowness.** PR 3345's last five heads each stopped printing at 98%, after `Tests/Workspaces/test_change_revert.py::test_revert_refuses_while_a_run_is_active`, and stayed silent until the cap. The last two runs set `faulthandler_timeout=90`, and its dump puts the main thread in the setup of that file's `turn` fixture, in `Tests/agent_runs_owner.py:40 owned_agent_runs_db`, with a worker thread inside `storage_admission._run` that is running a weakref finalizer (`Tests/agent_runs_owner.py:18 _release`) inside `real_profile_guard._hook` during `ctypes.CDLL.__init__`. `Tests/agent_runs_owner.py` is not on dev (it came with bb0b2cf6f8 on that branch); on dev the fixture builds `AgentRunsDB` directly. pytest's `--timeout=180` did not end it in the 14.5 and 19.5 silent minutes. No cap would have helped these five, and a higher one holds a hung job longer.
- **Six are slow runners.** Each cut log, scaled by the files it had finished against a complete log of the same list, would have needed 35.1 to 38.5 minutes. All six ran on Azure.

**Where the time goes** (job 139628967801, the slowest complete log: 4,966 tests, 167 files, 1,957 s of tests plus about 2 min of setup and collection):

| Folder | Seconds | Share | Tests |
| --- | --- | --- | --- |
| `Tests/UI` (three files) | 613 | 31% | 430 |
| `Tests/Chat` (27 files) | 482 | 25% | 495 |
| `Tests/CI` (44 files) | 421 | 22% | 2,011 |
| `Tests/Library` | 130 | 7% | 189 |
| the other 85 files | 311 | 16% | 1,841 |

One file is 28% of the lane: `Tests/UI/test_mcp_workbench.py` took 555 s (357 tests, about 1.6 s a test; its slowest take 6 to 7 s each). The next is `Tests/CI/test_console_p0_regression_gate.py` at 189 s, then the canvas profile preview file at 118 s and the settings defaults file at 78 s. `Tests/UI/test_mcp_tools_mode.py` adds 50 s. The workbench file runs in no other PR lane on dev (it is not in the UI census). It has been a rider since the lane's first commit, 063572bf53 (2026-08-29).

**Why runs differ: the hosted runner, as in TASK-42719, with a spread of 1.7 times.** Over the 81 jobs whose speed can be measured (75 complete logs and the 6 slow cut ones), each job's seconds over the files it shares with the reference log, divided by the reference's, run from 0.66 to 1.12 (median 0.91). The six AWS-hosted jobs ran at 0.66 to 0.72; the 75 Azure ones at 0.69 to 1.12, with `eastus` (median 1.00), `westcentralus` (1.01) and `westus` (0.98) slowest and `centralus` fastest (0.84). The image version made no difference inside Azure (medians 0.91 and 0.92). On the same 4,966-test list the fastest complete job (140208156941, AWS `us-east-1`) took 1,380 s of tests and the slowest (139628967801, Azure `westus`) 1,957 s: 1.42 times. Taking the 8 complete logs at 0.75 or faster against the 3 at 1.00 or slower, the 18 heaviest files took 1.20 to 1.69 times as long on the slow runners where all files together took 1.44: no file is slow by itself. A first comparison read 1.81 times because the fastest log did not run the 555 s workbench file at all; compare only files both logs ran. So the six slow cancellations are not tests that got slower. They are jobs that drew a runner 1.03 to 1.12 times as slow as the reference, on a list whose median job already takes 31 of 35 minutes.

**This job never sets the required check's wall clock.** In 335 full-run attempts since 2026-11-20 a UI shard finished last in 333; this job finished last in none. A longer cap costs no wait. A cancellation costs the author a manual rerun of the same 31 minutes.

**Options.** The table times the 81 measured runner speeds against today's list (the reference log's files), so it forecasts the next days rather than repeating the last ones: today the model gives median 31.3, matching the 31.2 observed. "Under the cap" is the slowest of the 81 against the cap; the house floor is 4 minutes (TASK-34986), and this task's AC #2 wants the p95 at least 5 under.

| | Change | Runners | Median | p90 | Slowest | Under the cap | Over the cap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | none (cap 35) | none | 31.3 | 34.6 | 38.5 | -3.5 | 6 of 81 |
| 1 | cap 35 to 40 | none | 31.3 | 34.6 | 38.5 | 1.5 | 0 |
| 2 | cap 35 to 44 | none | 31.3 | 34.6 | 38.5 | 5.5 | 0 |
| 3 | `Tests/CI` on 4 xdist workers, in its own step | none | 29.7 to 27.3 | 32.7 to 30.1 | 36.5 to 33.6 | -1.5 to 1.4 | 3 to 0 |
| 4 | the two MCP workbench files in a job of their own | +1 | 22.1 | 24.4 | 27.2 | 7.8 | 0 |
| 5 | the same two files into Console P0 (sandboxed) | none | 22.1 | 24.4 | 27.2 | 7.8 | 0 |

1. **Cap 40** leaves the slowest runner 1.5 minutes, below the floor.
2. **Cap 44.** It adds no runner and no runner time and cannot break a test. It is the highest cap that stays under the 45-minute busy-job deadline of the self-hosted runner controller (ADR-359; `CI_TEST_RUNNER` is unset today, so the hosted `ubuntu-latest` runs these jobs, but ADR-103's 2026-11-13 note keeps long lanes below that deadline). Costs: a hung job holds its runner 9 minutes longer; the lane's median stays at 31 minutes, and it has grown 0.34 minutes a day (29.1 to 31.5 over seven days), so 5.5 minutes of room lasts about two weeks. The job's own comment says "Add a job (a shard), not minutes".
3. **xdist for `Tests/CI`.** The folder is 421 s of this lane in the slowest log, and the most runner-sensitive part. The owner agreed on 2026-11-27 to run it on 4 workers in the light job (TASK-42628), and there the same folder took 158 s on a hosted runner (job 140738839181, 2,082 items; a public repository's runners have 4 vCPUs), against 266 to 421 s serially in this lane's logs. That would save 108 to 263 s here, depending on the runner's speed. Not measured in this lane (no Linux host here): it needs `Tests/CI` in a pytest step of its own, `pytest-xdist` in the install line, and the lane contracts in `Tests/CI` to follow. On its own it does not clear the floor; with option 4 it would take the slowest to about 22 minutes.
4. **A job of its own for the workbench pair.** `Tests/UI/test_mcp_workbench.py` and `Tests/UI/test_mcp_tools_mode.py` are 605 s of the lane's 1,957, so this lane drops to median 22.1, slowest 27.2, 7.8 under its cap with no change to the limit. The new job takes about 13 of its 20 minutes at the slowest runner (setup and collection about 1.5 minutes, 605 s times 1.12). Cost: one more runner per PR (37 concurrent test runners, 38 with perf-guard, against the account ceiling of 40 in ADR-103) and about 1.4 more runner-minutes of setup. It needs a new job, a verdict step in the aggregate and pins in several `Tests/CI` files (job order, verdict table, runner routing, lane tables, ADR-103's counts), and only a full CI run proves it, because the pair runs without the other riders for the first time. It is the shape the owner chose for TASK-34916 (2026-10-14), TASK-35575 (2026-10-19) and Console P0 (2026-11-24).
5. **Into Console P0 (sandboxed)** needs no new runner: that profile ran 90 times at median 6.9 and max 8.5 of its 35 minutes, so it would take about 20. Costs: the Console P0 lane table and sole-owner checks change, the job's name stops matching its contents, and the pair shares a process with that gate's files for the first time. No other lane has room: the nearest are `dreams-boundary` (max 24.5 of 35) and Console P0 `admission-sensitive 2` (max 20.2 of 35, a bootstrap profile).
6. **PR 3345** already carries the move (279 census node entries) and a rebalanced census, but it is an unmerged draft whose own PR Fast Lane has hung five times, and it adds four riders here. If it lands first, option 4 is not needed; if the owner wants this fixed first, taking the workbench change out of it is the alternative. Its edits to this job's block and to the `recovered-suites` block (TASK-42413) overlap any workflow change made for either, and whichever lands second must be rebased by hand.

**Recommendation: option 4.** One file is 28% of the lane, so moving it fixes the cause and leaves the cap alone, and it is the shape the owner chose for this lane's own admission-sensitive halves (TASK-34916, TASK-35575) and for Console P0 (2026-11-24). If a runner cannot be spared, option 2 is the smallest change, and option 5 the only move that needs no runner. The owner chose a cap raise instead for the recovered suites (2026-11-19, TASK-42413) and, per PR 3803's description, for the Dreams riders job (2026-11-28), so a cap is a fair answer here too, with the margin option 2 gives.

**Applied (2026-11-28): option 4.** Options 1, 2, 3, 5 and 6 are not. The one thing only CI shows is the pair running in a process of its own, without the other riders; the pull request's own full run is that evidence. AC #2 needs 20 full pull-request runs after it lands: this lane's slowest run well under 35 minutes (the model gives p90 24.4 and slowest 27.2 against its 35) and the new job's slowest well under its 20 (about 13 at the slowest runner seen). Measure with the tools in the session notes folder (`population.py` against a complete log of the new list) and the job times (`gh api repos/<owner>/<repo>/actions/runs/<id>/jobs`). PR 3345 also takes the workbench file out of this job's targets and adds four riders, so whichever lands second must be rebased by hand, and if it lands second its census entries would run the pair a second time.

**Verification of the applied change (2026-11-28; local Windows, Python 3.12.10, the private-profile runner).** `Tests/CI` on the edited tree: 54 failed, 2,036 passed, 16 skipped in 368 s. The 54 are the nodes that fail on untouched dev here (53 in `test_tiered_ci.py`, which PR 3802 addresses, and 1 symlink-privilege node in `test_canvas_mermaid_asset_checker.py`). A baseline run on the tree before the change had 61 failed: those 54 plus 7 Canvas Mermaid nodes that went green once three stale CRLF copies of `-text` files were restored. So nothing new fails. The six contract files that pin these jobs (`test_ci_queue_pressure_contract`, `test_derived_artifacts_workflow`, `test_console_p0_regression_gate`, `test_ui_lane_shard_count_statements`, `test_self_hosted_runner_routing`, `test_response_rules_critical_gate`): 1,077 passed, after the final edit. The new pins bite: removing the job from the aggregate's `needs`, dropping or weakening its verdict step, making the job `continue-on-error`, changing its cap, dropping one of its targets, shortening its `--timeout` or putting the workbench file back in `pr-fast-lane` each fails a contract (negative controls run against the workflow). `./scripts/preflight.sh` with Python 3.12.10: 12 of 13 checks pass; the 13th, the Canvas Mermaid assets, needs Python 3.12.11 exactly. The two files alone on this machine proved nothing either way: they are timing-sensitive mounted UI tests and flaked under load here (the first failing test, `test_open_add_server_form_gated_notifies_with_button_tooltip_copy`, passed once and failed once when run alone, with a `NoMatches` race), so the pull request's own full run is the evidence that they pass in a process of their own on Linux.

**Seen in the same data, not part of this task.**
- Lanes close to their own caps in the same runs (finished jobs since 2026-11-26: median, p90, max of the cap): `PR Fast Lane (admission-sensitive 4)` 25.3, 31.2, 34.9 of 35 and no cancellation yet; Console P0 `admission-sensitive` 23.7, 26.8, 34.2 of 35; `PR Fast Lane (admission-sensitive 3)` 32.4 max of 35; the UI shards 43 of about 3,280 cancelled at 20 minutes. The same method applies.
- PR 3345's `.github/workflows/derived-artifacts.yml` differs from dev in two caps it does not mention: `dev-light` 30 to 15 (commit 34a601690c) and `plugins-recovery-ui` 25 to 15 (c3c8ba351d, which also deletes the owner's 2026-11-26 decision comment). Dev's tip is an ancestor of that branch, so these read as old values kept in a conflict resolution.

**How it was measured.** Job records and annotations from the GitHub API; per-file times from 86 job logs of this job (a progress line is stamped when its newline arrives, so a file's time is the gap to the line before it, and the first test of a file is booked to the file before it); runner speed from the files a log shares with the reference at 1 s or more, dropping files whose ratio is outside 0.4 to 2.5 times the log's median (a pull request changed them); a cut log's total from its finished files. The tools, the per-log speed table and this note's tables are in the session's notes folder (`ci-lane-cap-tools/`, `ci-lane-cap-data/`).
<!-- SECTION:NOTES:END -->
