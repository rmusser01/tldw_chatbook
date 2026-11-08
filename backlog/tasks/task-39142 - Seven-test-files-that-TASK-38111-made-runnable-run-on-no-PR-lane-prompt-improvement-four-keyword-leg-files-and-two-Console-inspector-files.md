---
id: TASK-39142
title: "Seven test files that TASK-38111 made runnable run on no PR lane: prompt improvement, four keyword-leg files and two Console inspector files"
status: In Progress
assignee: ['@claude']
created_date: 2026-11-07 02:03
updated_date: 2026-11-08 06:45
labels:
- follow-up
- ci
- tests
- needs-owner-decision
priority: medium
references:
- .github/workflows/derived-artifacts.yml
- Tests/CI/test_ci_queue_pressure_contract.py
- Tests/CI/test_response_rules_critical_gate.py
- scripts/ui_pr_gate_census.txt
- Tests/Prompt_Management/test_prompt_improvement_service.py
- Tests/RAG_Search/test_keyword_leg_chacha.py
- Tests/RAG_Search/test_keyword_leg_prompts.py
- Tests/RAG_Search/test_keyword_leg_pushdown.py
- Tests/RAG_Search/test_keyword_leg_tiered_merge.py
- Tests/UI/test_console_inspector_focus_carriers.py
- Tests/UI/test_console_inspector_workflow.py
dependencies:
- TASK-38111
- TASK-39049
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 11 of the 2026-10-23 burn-down; checked on dev 2684e5453f, 2026-11-07, with TASK-38111's run logs (round-11 branch fu/r11-recovery).

**The seven files.**
- `Tests/Prompt_Management/test_prompt_improvement_service.py`;
- four `Tests/RAG_Search/test_keyword_leg_*` files: chacha, prompts, pushdown, tiered_merge;
- `Tests/UI/test_console_inspector_focus_carriers.py`;
- `Tests/UI/test_console_inspector_workflow.py`.

None is in the census or any PR workflow. The fifth keyword-leg file, `test_keyword_leg_db_resolution`, is TASK-39049's AC#3. 36e0f56c7c greened `test_console_inspector_workflow` on dev, not TASK-38111.

**Local times after TASK-38111.** These come from its scratch logs, not its notes, and the load was not recorded.

| File | `-n 4` | Serial, no guard |
|---|---|---|
| prompt improvement | 21.4 s | 22.3 s |
| keyword-leg chacha | 51.7 s | 31.5 s |
| keyword-leg prompts | 67.2 s | 14.2 s |
| keyword-leg pushdown | 67.3 s | 45.9 s |
| keyword-leg tiered_merge | 61.3 s | 68.6 s |
| inspector focus carriers | 115.5 s | 300.1 s (15 tests) |
| inspector workflow | 57.8 s | not run |

**Lane budgets.**
- Three PR runs on 2026-11-06 (42306963606, 42309351181, 42310788537): the Dreams boundary step took 6:56-7:34 and its job 11:03-12:05 of 20 min.
- TASK-39049's notes put the job at about 13.2-16.8 min once round 11 lands.
- No file's behaviour under `--storage-lease-guard` is known.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The owner decides the lane for the five non-UI files (a new derived-artifacts job or an existing one); the decision is recorded
- [x] #2 Each of the seven files runs on a PR lane, never the fast PR contract; the two Tests/UI files use UI census tail entries of about 30 s each, split by node where needed
- [ ] #3 Every step touched is timed 3 times with its exact command, with the load, and the CI job times of every touched lane are quoted before and after
- [x] #4 Any target added to the Dreams boundary step passes under `--storage-lease-guard`, sits after the pinned storage-pausing targets, and leaves the job under its 20 min cap with the margin recorded; all lane pins are updated
- [x] #5 The keyword-leg files end on one lane together with TASK-39049's `test_keyword_leg_db_resolution`, or the notes say why not
<!-- AC:END -->

## Implementation Plan

1. Read recent CI times of the lanes touched (read-only `gh`).
2. Add the owner's new derived-artifacts job, "PR lane (recovered suites)" (`recovered-suites`), with the five non-UI files, serial under `--storage-lease-guard`, after the sandboxed suites (bootstrap_profile files last, TASK-32873); aggregation verdict, `Tests/CI` pins and ADR-103 counts.
3. Put the two Tests/UI files on UI census tail entries of about 30 s each, split by node where a whole file is heavier, measured with the UI step's command.
4. Time the new step 3 times with `jobstep.sh`; keep the keyword-leg files together with TASK-39049's `test_keyword_leg_db_resolution`.

## Implementation Notes

**Owner decision (2026-11-06):** "New CI job: add one derived-artifacts job, 'PR lane (recovered suites)', holding all of these files serially under --storage-lease-guard; one more runner per PR; ADR-103's job and runner counts updated. The Dreams job keeps its margin; only TASK-39043's two small files go there."

**The five non-UI files (AC#1, AC#5).** `test_prompt_improvement_service.py` and the four keyword-leg files run in the new `recovered-suites` job, with TASK-39049's `test_keyword_leg_db_resolution.py` beside them, so all five keyword-leg files share one lane. They are bootstrap_profile files (TASK-38111), so they follow every sandboxed target in the step (TASK-32873). The job has the same install line, serial pytest, `--timeout=180`, `timeout-minutes: 20`, its own verdict step in the required check, and its argv, run-once rule and aggregation are pinned in `Tests/CI`; ADR-103 says 27 runners in parallel (28 with perf-guard).

**Guard findings, fixed in test code.** Under `--storage-lease-guard` 33 keyword-leg tests failed at teardown: their `RAGService` was never closed, so the FTS5 pool thread kept a `tldw_cli_media_v2.db` lease. `Tests/RAG_Search/conftest.py` now retires every pool (`close_all_pools()`) after each test in a `test_keyword_leg_*` file. Red: 33 lease errors in the step's first run; green: 0.

**The two Tests/UI files (AC#2).**
- `test_console_inspector_workflow.py` is on the UI census tail: one entry per test, 18 entries, 1.6-12.3 s each serially at load 50-155 (69 s whole).
- `test_console_inspector_focus_carriers.py` (module `pytestmark = pytest.mark.bootstrap_profile`, TASK-38111) is on the census tail by node, 15 entries. Each node, serially with the UI step's command in two runs (load 50-155 and about 160): 0.9-20.1 s and 0.0-27.5 s; the heaviest, `test_every_rail_tab_stop_changes_what_it_paints_on_focus[size1]`, 18.7 and 27.5 s. It first ran in the recovered-suites step because I wrongly read the census as unable to hold bootstrap_profile files; review r12l-rev1 corrected that (512 census entries already were).
- Spot checks of the new deal in lane order: UI shard 20 (holds the heaviest focus-carriers node) 107 passed in 955 s at load 39-58; shard 34 (holds the full config file) 109 passed in 895 s at load 36-58.

**AC#3 (timings met, after times open).**

**The recovered-suites step**, `jobstep.sh fu-r12-lanes "Run the recovered suites"` (14 targets, serial, `--storage-lease-guard --timeout=180`), 3 runs on 2026-11-07: 534.8 s (load 40-63), 703.2 s (load 40-60), 641.7 s (load 50-60). Each run: 723 passed, 1 skipped, 2 failed. The 2 are `test_executor_failure_uses_stable_job_terminal[artifact_incompatible-failed]` and `[engine_crashed-failed]` in the ingest runner, TASK-39140's stale `error_detail` pin, red on dev too and fixed on round-12 branch fu/r12-ctlr (489e2338f5); the step is green once both land. No lease finding in any run. The slowest run plus about 1.5 min of setup is about 13.2 min, 6.8 min under the job's 20-min cap; CI ran comparable lanes at 0.5-1.0 times this machine's loaded time.

The keyword-leg files alone, serially under the guard: 339 passed with the two prompt files in 171.8 s. **CI before (read-only `gh`).** Dreams boundary job over 15 PR runs, 2026-11-06 14:36Z to 2026-11-07 00:55Z (42235403758 to 42321107301): job 9.80-14.05 min, step 6.10-8.93 min. None of them has round 11's three Dreams additions; no PR run of round-11 dev had finished when these were read (the queue held 04:33Z-05:31Z runs, all queued or cancelled). The recovered-suites job is new, so it has no before time. CI after times need a PR run of this branch, so AC#3 stays open.

**AC#4.** No target of this task went to the Dreams step. The recovered-suites step also runs under `--storage-lease-guard` and holds no storage-pausing target; its margin is above.

**Files.** `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`, `Tests/CI/test_derived_artifacts_workflow.py`, `Tests/CI/test_ui_lane_shard_count_statements.py`, `Tests/RAG_Search/conftest.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`, ADR-103.

**Review r12l-rev1 (2026-11-08).** The two bootstrap_profile Tests/UI files left the recovered-suites step for the UI census (the census does hold bootstrap_profile files: 512 entries already, and TASK-36478 restores the shared bootstrap config after each test; my earlier "may not hold" was wrong). TASK-39124's built-in pixel buddy file joined the step last. The final step, 13 targets, `jobstep.sh fu-r12-lanes "Run the recovered suites"`, 3 runs: 363.3 s (load 38 falling to 21), 742.9 s (load 21-50), 467.1 s (load 50-43); each 715 passed, 1 skipped, 2 failed (TASK-39140's pin, fixed on fu/r12-ctlr), no lease finding. The slowest run plus about 1.5 min of setup is about 13.9 min, 6.1 under the cap.

**After the round-12 restack (2026-11-08).** The step now has 20 targets (TASK-39124's pixel buddy file, TASK-39119's six reply-pin nodes and the goal recovery file joined it). `jobstep.sh fu-r12-lanes "Run the recovered suites"`, 3 runs: 534.7 s (load 22-33), 353.8 s (load 22-28), 312.5 s (load 20-28); each 733 passed, 1 skipped, nothing red (TASK-39140's pin is fixed in the stack) and no lease finding. The slowest run plus about 1.5 min of setup is about 10.4 of the job's 20 min.
